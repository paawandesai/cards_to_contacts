"""
AI-Powered Business Card Reader with GPT Integration
A Streamlit application for extracting contact information from business cards
"""

import streamlit as st
import pandas as pd
import io
from datetime import datetime
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

# Check if we're in embed mode
embed_mode = st.query_params.get("embed", "false").lower() == "true"

# Page configuration
st.set_page_config(
    page_title="AI Business Card Reader",
    page_icon="🪪",
    layout="wide",
    initial_sidebar_state="expanded" if not embed_mode else "collapsed"
)

# Initialize session state
if 'total_cost' not in st.session_state:
    st.session_state.total_cost = 0.0
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = pd.DataFrame()
if 'api_key_validated' not in st.session_state:
    st.session_state.api_key_validated = False
if 'notion_validated' not in st.session_state:
    st.session_state.notion_validated = False

# Apply embed mode styling
if embed_mode:
    st.markdown("""
    <style>
    .stApp > header {
        background-color: transparent;
    }
    
    .stApp {
        margin-top: -80px;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stDeployButton {display:none;}
    .stDecoration {display:none;}
    
    .stApp > div:first-child {
        margin-top: -20px;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    # Header
    if embed_mode:
        st.markdown("### 🪪 AI Business Card Reader")
        st.markdown("*Extract contact information from business cards using GPT Vision*")
    else:
        st.title("🪪 AI Business Card Reader")
        st.markdown("Extract contact information from business cards using GPT Vision")
    
    # Sidebar for API Configuration
    with st.sidebar:
        st.header("🔐 API Configuration")
        
        # OpenAI API Key (Required)
        st.subheader("OpenAI API Key (Required)")
        openai_key = st.text_input(
            "Enter your OpenAI API Key",
            type="password",
            placeholder="sk-proj-...",
            help="Get your API key from https://platform.openai.com/api-keys"
        )
        
        # Add tier information
        with st.expander("ℹ️ About API Key Tiers"):
            st.markdown("""
            **Free Tier** (New Keys):
            - 3 requests/minute, 200 requests/day
            - Good for testing, limited for production
            
            **Tier 1** (After $5 payment):
            - 500 requests/minute, 10K requests/day
            - Recommended for regular use
            
            **Higher Tiers** (Tier 2+):
            - Even higher limits based on usage history
            
            **Check your current tier:** https://platform.openai.com/account/limits
            **Upgrade billing:** https://platform.openai.com/account/billing/overview
            """)
        
        # Validate OpenAI API key
        if openai_key:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔍 Validate OpenAI Key"):
                    with st.spinner("Validating API key..."):
                        if validate_api_key(openai_key):
                            st.session_state.api_key_validated = True
                            st.success("✅ API key is valid!")
                        else:
                            st.session_state.api_key_validated = False
                            st.error("❌ Invalid API key")
            
            with col2:
                if st.button("⏭️ Skip Validation"):
                    st.session_state.api_key_validated = True
                    st.warning("⚠️ Validation skipped - proceeding with unvalidated key")
                    st.info("💡 If the key is invalid, processing will fail")
            
            # Development mode toggle
            dev_mode = st.checkbox(
                "🔧 Development Mode",
                help="Skip validation automatically for development/testing"
            )
            
            if dev_mode and openai_key and not st.session_state.api_key_validated:
                st.session_state.api_key_validated = True
                st.info("🔧 Development mode: API key validation skipped")
        
        # Notion Integration (Optional)
        st.subheader("🗂️ Notion Integration (Optional)")
        with st.expander("Configure Notion Integration"):
            notion_token = st.text_input(
                "Notion Integration Token",
                type="password",
                placeholder="secret_...",
                help="Create an integration at https://www.notion.so/my-integrations"
            )
            
            notion_database_id = st.text_input(
                "Notion Database ID",
                placeholder="32-character string",
                help="Find this in your database URL"
            )
            
            # Validate Notion credentials
            if notion_token and notion_database_id:
                if st.button("🔍 Test Notion Connection"):
                    with st.spinner("Testing Notion connection..."):
                        result = validate_notion_credentials(notion_token, notion_database_id)
                        if result["valid"]:
                            st.session_state.notion_validated = True
                            st.success(f"✅ Connected to: {result['database_title']}")
                        else:
                            st.session_state.notion_validated = False
                            st.error(f"❌ {result['error']}")
        
        # Privacy Notice
        st.info("🔒 API keys are only stored in your browser session and never saved permanently.")
        
        # Cost Display
        st.subheader("💰 Session Cost")
        st.metric("Total Cost", format_cost(st.session_state.total_cost))
        
        if st.button("Clear Session Cost"):
            st.session_state.total_cost = 0.0
            st.rerun()
    
    # Main Content
    if not openai_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to get started.")
        st.markdown("""
        ### How to get your OpenAI API Key:
        1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
        2. Sign in to your account
        3. Click "Create new secret key"
        4. Copy the key and paste it in the sidebar
        
        ### Features:
        - 🔍 Extract contact information from business cards
        - 📊 Process multiple cards at once
        - 💰 Real-time cost tracking
        - 📄 Export to CSV/Excel
        - 🗂️ Optional Notion integration
        - 📱 Mobile-friendly interface
        """)
        return
    
    # Model Selection
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_model = st.selectbox(
            "Select GPT Model",
            options=list(MODEL_OPTIONS.keys()),
            format_func=lambda x: MODEL_OPTIONS[x]["name"],
            help="Choose between cost-effective and high-performance models"
        )
    
    with col2:
        model_info = MODEL_OPTIONS[selected_model]
        st.metric("Cost per Image", format_cost(model_info["vision_cost"]))
    
    # Model Information
    st.info(f"📋 {model_info['description']}")
    
    # File Upload
    st.subheader("📤 Upload Business Card Images")
    uploaded_files = st.file_uploader(
        "Choose business card images",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload one or more business card images"
    )
    
    if uploaded_files:
        # Display uploaded images
        st.subheader("📷 Uploaded Images")
        cols = st.columns(min(len(uploaded_files), 4))
        for idx, file in enumerate(uploaded_files):
            with cols[idx % 4]:
                st.image(file, caption=f"Image {idx+1}", use_column_width=True)
        
        # Cost Estimation
        num_images = len(uploaded_files)
        estimated_cost = calculate_estimated_cost(selected_model, num_images)
        
        st.info(f"💰 Estimated cost for {num_images} image(s): {format_cost(estimated_cost)}")
        
        # Process Button
        if st.button("🚀 Extract Business Cards", type="primary"):
            if not st.session_state.api_key_validated:
                st.warning("⚠️ Please validate your OpenAI API key first")
                return
            
            process_images(uploaded_files, selected_model, openai_key)
    
    # Display Results
    if not st.session_state.extracted_data.empty:
        display_results(notion_token if 'notion_token' in locals() else None, 
                       notion_database_id if 'notion_database_id' in locals() else None)

def process_images(files, model, api_key):
    """Process uploaded images and extract business card data"""
    
    all_cards = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, file in enumerate(files):
        status_text.text(f"Processing image {idx + 1} of {len(files)}...")
        
        # Reset file pointer
        file.seek(0)
        
        # Extract business cards
        result = extract_business_cards(file, model, api_key)
        
        if "error" in result:
            st.error(f"Error processing {file.name}: {result['error']}")
            continue
        
        # Add cards to collection
        cards = result.get("cards", [])
        for card in cards:
            card["source_image"] = file.name
        
        all_cards.extend(cards)
        
        # Update usage
        if "usage" in result:
            usage = result["usage"]
            total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            total_usage["total_tokens"] += usage.get("total_tokens", 0)
        
        # Update progress
        progress_bar.progress((idx + 1) / len(files))
    
    # Calculate actual cost
    actual_cost = calculate_actual_cost(total_usage, model)
    vision_cost = len(files) * MODEL_OPTIONS[model]["vision_cost"]
    total_cost = actual_cost + vision_cost
    
    # Update session cost
    st.session_state.total_cost += total_cost
    
    # Process extracted data
    if all_cards:
        df = process_extracted_data(all_cards)
        df = detect_duplicates(df)
        st.session_state.extracted_data = df
        
        # Show results
        st.success(f"✅ Extracted {len(all_cards)} business cards from {len(files)} images")
        st.info(f"💰 Actual cost: {format_cost(total_cost)}")
        
        # Show validation warnings
        warnings = validate_data(df)
        if any(warnings.values()):
            st.warning("⚠️ Data validation warnings:")
            for warning_type, messages in warnings.items():
                if messages:
                    st.write(f"**{warning_type.replace('_', ' ').title()}:**")
                    for msg in messages:
                        st.write(f"- {msg}")
    else:
        st.error("❌ No business cards found in the uploaded images")
    
    # Clear progress
    progress_bar.empty()
    status_text.empty()

def display_results(notion_token, notion_database_id):
    """Display extracted data in an editable format"""
    
    st.subheader("📊 Extracted Business Card Data")
    
    # Data editor
    edited_df = st.data_editor(
        st.session_state.extracted_data,
        column_config={
            "confidence": st.column_config.ProgressColumn(
                "Confidence",
                help="Extraction confidence score",
                min_value=0,
                max_value=1,
                format="%.2f"
            ),
            "verified": st.column_config.CheckboxColumn(
                "Verified",
                help="Mark as manually verified",
                default=False
            ),
            "is_duplicate": st.column_config.CheckboxColumn(
                "Duplicate",
                help="Potential duplicate entry",
                default=False
            ),
            "card_number": st.column_config.NumberColumn(
                "Card #",
                help="Card number",
                min_value=1
            )
        },
        hide_index=True,
        use_container_width=True,
        key="business_card_editor"
    )
    
    # Update session data
    st.session_state.extracted_data = edited_df
    
    # Export section
    st.subheader("📤 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV Export
        csv_data = export_to_csv(edited_df)
        st.download_button(
            label="📄 Download CSV",
            data=csv_data,
            file_name=generate_filename("business_cards", "csv"),
            mime="text/csv",
            help="Download data as CSV file"
        )
    
    with col2:
        # Excel Export
        excel_data = export_to_excel(edited_df)
        st.download_button(
            label="📊 Download Excel",
            data=excel_data,
            file_name=generate_filename("business_cards", "xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download data as Excel file with summary"
        )
    
    with col3:
        # Notion Upload
        if notion_token and notion_database_id and st.session_state.notion_validated:
            if st.button("📝 Send to Notion", type="secondary"):
                upload_to_notion_database(edited_df, notion_token, notion_database_id)
        else:
            st.button("📝 Send to Notion", disabled=True, help="Configure Notion integration first")
    
    # Data summary
    st.subheader("📈 Data Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cards", len(edited_df))
    
    with col2:
        high_confidence = len(edited_df[edited_df['confidence'] > 0.8])
        st.metric("High Confidence", high_confidence)
    
    with col3:
        with_email = len(edited_df[edited_df['email'].str.len() > 0])
        st.metric("With Email", with_email)
    
    with col4:
        duplicates = len(edited_df[edited_df['is_duplicate'] == True])
        st.metric("Duplicates", duplicates)

def upload_to_notion_database(df, notion_token, notion_database_id):
    """Upload data to Notion database"""
    
    with st.spinner("Uploading to Notion..."):
        result = upload_to_notion(df, notion_token, notion_database_id)
        
        if result["success"]:
            results = result["results"]
            st.success(f"✅ Successfully uploaded {results['success']} contacts to Notion database: {result['database_title']}")
            
            if results["failed"] > 0:
                st.warning(f"⚠️ {results['failed']} entries failed to upload")
                with st.expander("View errors"):
                    for error in results["errors"]:
                        st.write(f"- {error}")
        else:
            st.error(f"❌ Failed to upload to Notion: {result['error']}")

if __name__ == "__main__":
    main()