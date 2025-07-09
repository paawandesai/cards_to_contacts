"""
Data processing utilities for business card extraction
"""

import pandas as pd
import streamlit as st
from typing import List, Dict, Any
import re
import io
from datetime import datetime

def process_extracted_data(cards_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Process extracted business card data into a clean DataFrame
    
    Args:
        cards_data: List of extracted business card data
        
    Returns:
        pandas DataFrame with cleaned and processed data
    """
    processed_data = []
    
    for i, card in enumerate(cards_data):
        card_data = card.get("extracted_data", {})
        
        # Create a clean record
        record = {
            "card_number": card.get("card_number", i + 1),
            "name": clean_text(card_data.get("name", "")),
            "title": clean_text(card_data.get("title", "")),
            "company": clean_text(card_data.get("company", "")),
            "email": clean_email(card_data.get("email", "")),
            "phone": clean_phone(card_data.get("phone", "")),
            "website": clean_url(card_data.get("website", "")),
            "address": clean_text(card_data.get("address", "")),
            "linkedin": clean_url(card_data.get("linkedin", "")),
            "additional_notes": clean_text(card_data.get("additional_notes", "")),
            "confidence": card.get("confidence", 0.0),
            "verified": False  # User can check this manually
        }
        
        processed_data.append(record)
    
    return pd.DataFrame(processed_data)

def clean_text(text: str) -> str:
    """Clean and normalize text data"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common OCR artifacts
    text = text.replace('|', '').replace('_', '').replace('^', '')
    
    return text

def clean_email(email: str) -> str:
    """Clean and validate email addresses"""
    if not email:
        return ""
    
    # Extract email using regex
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    matches = re.findall(email_pattern, email)
    
    if matches:
        return matches[0].lower()
    
    return clean_text(email)

def clean_phone(phone: str) -> str:
    """Clean and format phone numbers"""
    if not phone:
        return ""
    
    # Remove all non-digit characters except + and spaces
    phone = re.sub(r'[^\d\+\s\(\)\-\.]', '', phone)
    
    # Remove extra whitespace
    phone = re.sub(r'\s+', ' ', phone).strip()
    
    return phone

def clean_url(url: str) -> str:
    """Clean and validate URLs"""
    if not url:
        return ""
    
    url = clean_text(url)
    
    # Add https:// if missing
    if url and not url.startswith(('http://', 'https://')):
        if '.' in url:
            url = f"https://{url}"
    
    return url

def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect potential duplicate entries based on name, email, or phone
    
    Args:
        df: DataFrame with business card data
        
    Returns:
        DataFrame with duplicate flags
    """
    df = df.copy()
    df['is_duplicate'] = False
    
    # Check for duplicates based on name (if not empty)
    name_mask = df['name'].str.len() > 0
    if name_mask.any():
        df.loc[name_mask, 'is_duplicate'] = df.loc[name_mask, 'name'].duplicated(keep=False)
    
    # Check for duplicates based on email (if not empty)
    email_mask = df['email'].str.len() > 0
    if email_mask.any():
        email_duplicates = df.loc[email_mask, 'email'].duplicated(keep=False)
        df.loc[email_mask, 'is_duplicate'] = df.loc[email_mask, 'is_duplicate'] | email_duplicates
    
    # Check for duplicates based on phone (if not empty)
    phone_mask = df['phone'].str.len() > 0
    if phone_mask.any():
        phone_duplicates = df.loc[phone_mask, 'phone'].duplicated(keep=False)
        df.loc[phone_mask, 'is_duplicate'] = df.loc[phone_mask, 'is_duplicate'] | phone_duplicates
    
    return df

def validate_data(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Validate business card data and return warnings
    
    Args:
        df: DataFrame with business card data
        
    Returns:
        Dictionary with validation warnings
    """
    warnings = {
        "empty_fields": [],
        "invalid_emails": [],
        "low_confidence": [],
        "missing_key_info": []
    }
    
    for index, row in df.iterrows():
        # Check for empty key fields
        if not row['name'] and not row['company']:
            warnings["missing_key_info"].append(f"Row {index + 1}: Missing both name and company")
        
        # Check email format
        if row['email'] and '@' not in row['email']:
            warnings["invalid_emails"].append(f"Row {index + 1}: Invalid email format")
        
        # Check confidence score
        if row['confidence'] < 0.7:
            warnings["low_confidence"].append(f"Row {index + 1}: Low confidence ({row['confidence']:.2f})")
        
        # Check for empty critical fields
        empty_fields = []
        if not row['name']:
            empty_fields.append('name')
        if not row['email'] and not row['phone']:
            empty_fields.append('contact info')
        
        if empty_fields:
            warnings["empty_fields"].append(f"Row {index + 1}: Missing {', '.join(empty_fields)}")
    
    return warnings

def export_to_csv(df: pd.DataFrame) -> str:
    """Export DataFrame to CSV string"""
    return df.to_csv(index=False)

def export_to_excel(df: pd.DataFrame) -> bytes:
    """Export DataFrame to Excel bytes"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Main data sheet
        df.to_excel(writer, sheet_name='Business Cards', index=False)
        
        # Summary sheet
        summary_data = {
            'Metric': [
                'Total Cards Extracted',
                'High Confidence Cards (>0.8)',
                'Medium Confidence Cards (0.6-0.8)',
                'Low Confidence Cards (<0.6)',
                'Cards with Email',
                'Cards with Phone',
                'Cards with Website'
            ],
            'Count': [
                len(df),
                len(df[df['confidence'] > 0.8]),
                len(df[(df['confidence'] >= 0.6) & (df['confidence'] <= 0.8)]),
                len(df[df['confidence'] < 0.6]),
                len(df[df['email'].str.len() > 0]),
                len(df[df['phone'].str.len() > 0]),
                len(df[df['website'].str.len() > 0])
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    output.seek(0)
    return output.getvalue()

def generate_filename(prefix: str, extension: str) -> str:
    """Generate filename with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"