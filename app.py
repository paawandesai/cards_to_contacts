"""
Flask-based AI Business Card Reader
A clean, professional web application for extracting contact information from business cards
"""

import os
import io
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, url_for
import pandas as pd

# Import existing business logic
from config.models import MODEL_OPTIONS, calculate_estimated_cost, format_cost
from utils.gpt_vision import extract_business_cards, calculate_actual_cost, validate_api_key
from utils.data_processing import (
    process_extracted_data, 
    detect_duplicates, 
    validate_data,
    export_to_csv,
    export_to_excel,
    generate_filename
)
from utils.notion_client import upload_to_notion, validate_notion_credentials

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global session data (in production, use Redis or database)
session_data = {}

@app.route('/')
def index():
    """Main page - business card upload and processing"""
    embed_mode = request.args.get('embed', 'false').lower() == 'true'
    return render_template('index.html', 
                         embed_mode=embed_mode,
                         model_options=MODEL_OPTIONS)

@app.route('/validate_api_key', methods=['POST'])
def validate_openai_key():
    """Validate OpenAI API key"""
    data = request.get_json()
    api_key = data.get('api_key')
    
    if not api_key:
        return jsonify({'valid': False, 'error': 'API key is required'})
    
    try:
        is_valid = validate_api_key(api_key)
        return jsonify({'valid': is_valid})
    except Exception as e:
        return jsonify({'valid': False, 'error': str(e)})

@app.route('/validate_notion', methods=['POST'])
def validate_notion():
    """Validate Notion credentials"""
    data = request.get_json()
    notion_token = data.get('notion_token')
    notion_database_id = data.get('notion_database_id')
    
    if not notion_token or not notion_database_id:
        return jsonify({'valid': False, 'error': 'Both token and database ID are required'})
    
    try:
        result = validate_notion_credentials(notion_token, notion_database_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'valid': False, 'error': str(e)})

@app.route('/process_cards', methods=['POST'])
def process_business_cards():
    """Process uploaded business card images"""
    try:
        # Get form data
        api_key = request.form.get('api_key')
        model = request.form.get('model', 'gpt-4o-mini')
        files = request.files.getlist('images')
        
        if not api_key:
            return jsonify({'error': 'OpenAI API key is required'}), 400
        
        if not files or files[0].filename == '':
            return jsonify({'error': 'Please upload at least one image'}), 400
        
        # Process images
        all_cards = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        for idx, file in enumerate(files):
            if file and file.filename:
                # Reset file pointer
                file.seek(0)
                
                # Extract business cards
                result = extract_business_cards(file, model, api_key)
                
                if "error" in result:
                    return jsonify({'error': f'Error processing {file.filename}: {result["error"]}'}), 400
                
                # Add cards to collection
                cards = result.get("cards", [])
                for card in cards:
                    card["source_image"] = file.filename
                
                all_cards.extend(cards)
                
                # Update usage
                if "usage" in result:
                    usage = result["usage"]
                    total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                    total_usage["total_tokens"] += usage.get("total_tokens", 0)
        
        # Calculate costs
        actual_cost = calculate_actual_cost(total_usage, model)
        vision_cost = len(files) * MODEL_OPTIONS[model]["vision_cost"]
        total_cost = actual_cost + vision_cost
        
        # Process extracted data
        if all_cards:
            df = process_extracted_data(all_cards)
            df = detect_duplicates(df)
            
            # Store in session (use session ID in production)
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            session_data[session_id] = {
                'dataframe': df,
                'total_cost': total_cost,
                'cards_count': len(all_cards),
                'images_count': len(files)
            }
            
            # Validate data and get warnings
            warnings = validate_data(df)
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'cards_extracted': len(all_cards),
                'images_processed': len(files),
                'total_cost': format_cost(total_cost),
                'warnings': warnings,
                'data': df.to_dict('records')
            })
        else:
            return jsonify({'error': 'No business cards found in the uploaded images'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/export/<format>/<session_id>')
def export_data(format, session_id):
    """Export processed data as CSV or Excel"""
    if session_id not in session_data:
        return jsonify({'error': 'Session not found'}), 404
    
    df = session_data[session_id]['dataframe']
    
    try:
        if format.lower() == 'csv':
            csv_data = export_to_csv(df)
            filename = generate_filename("business_cards", "csv")
            
            return send_file(
                io.BytesIO(csv_data.encode()),
                mimetype='text/csv',
                as_attachment=True,
                download_name=filename
            )
            
        elif format.lower() == 'excel':
            excel_data = export_to_excel(df)
            filename = generate_filename("business_cards", "xlsx")
            
            return send_file(
                io.BytesIO(excel_data),
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name=filename
            )
        else:
            return jsonify({'error': 'Invalid format. Use csv or excel'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

@app.route('/upload_to_notion', methods=['POST'])
def upload_to_notion_database():
    """Upload data to Notion database"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        notion_token = data.get('notion_token')
        notion_database_id = data.get('notion_database_id')
        
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404
        
        if not notion_token or not notion_database_id:
            return jsonify({'error': 'Notion credentials are required'}), 400
        
        df = session_data[session_id]['dataframe']
        result = upload_to_notion(df, notion_token, notion_database_id)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Notion upload failed: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for AWS App Runner"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # For local development
    app.run(debug=True, host='0.0.0.0', port=8080)