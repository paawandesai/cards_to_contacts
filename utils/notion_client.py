"""
Notion API integration for business card data
"""

import requests
import streamlit as st
import pandas as pd
from typing import Dict, Any, List
import json

class NotionClient:
    def __init__(self, token: str, database_id: str):
        self.token = token
        self.database_id = database_id
        self.base_url = "https://api.notion.com/v1"
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28"
        }
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection to Notion API and database"""
        try:
            # Test API connection
            response = requests.get(f"{self.base_url}/users/me", headers=self.headers)
            if response.status_code != 200:
                return {"success": False, "error": "Invalid Notion token"}
            
            # Test database access
            response = requests.get(f"{self.base_url}/databases/{self.database_id}", headers=self.headers)
            if response.status_code != 200:
                return {"success": False, "error": "Cannot access database. Check database ID and permissions."}
            
            database_info = response.json()
            return {
                "success": True,
                "database_title": database_info.get("title", [{}])[0].get("plain_text", "Unknown"),
                "database_id": self.database_id
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_database_properties(self) -> Dict[str, Any]:
        """Get database properties to understand structure"""
        try:
            response = requests.get(f"{self.base_url}/databases/{self.database_id}", headers=self.headers)
            if response.status_code == 200:
                return response.json().get("properties", {})
            return {}
        except Exception:
            return {}
    
    def create_page(self, card_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new page in the Notion database"""
        try:
            # Map business card data to Notion properties
            properties = self._map_card_to_notion_properties(card_data)
            
            data = {
                "parent": {"database_id": self.database_id},
                "properties": properties
            }
            
            response = requests.post(f"{self.base_url}/pages", headers=self.headers, json=data)
            
            if response.status_code == 200:
                return {"success": True, "page_id": response.json().get("id")}
            else:
                error_msg = response.json().get("message", "Unknown error")
                return {"success": False, "error": error_msg}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _map_card_to_notion_properties(self, card_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map business card data to Notion database properties"""
        properties = {}
        
        # Standard property mappings
        property_mappings = {
            "name": ("Name", "title"),
            "title": ("Title", "rich_text"),
            "company": ("Company", "rich_text"),
            "email": ("Email", "email"),
            "phone": ("Phone", "phone_number"),
            "website": ("Website", "url"),
            "address": ("Address", "rich_text"),
            "linkedin": ("LinkedIn", "url"),
            "additional_notes": ("Notes", "rich_text")
        }
        
        for card_field, (notion_field, notion_type) in property_mappings.items():
            value = card_data.get(card_field, "")
            if value:  # Only add non-empty values
                properties[notion_field] = self._format_notion_property(value, notion_type)
        
        # Add confidence score as a number
        if "confidence" in card_data:
            properties["Confidence"] = {
                "number": round(card_data["confidence"], 2)
            }
        
        # Add extraction date
        properties["Extracted Date"] = {
            "date": {
                "start": pd.Timestamp.now().isoformat()
            }
        }
        
        return properties
    
    def _format_notion_property(self, value: str, property_type: str) -> Dict[str, Any]:
        """Format value according to Notion property type"""
        if property_type == "title":
            return {"title": [{"text": {"content": str(value)[:2000]}}]}  # Limit to 2000 chars
        elif property_type == "rich_text":
            return {"rich_text": [{"text": {"content": str(value)[:2000]}}]}
        elif property_type == "email":
            return {"email": str(value) if "@" in str(value) else None}
        elif property_type == "phone_number":
            return {"phone_number": str(value)}
        elif property_type == "url":
            # Ensure URL has protocol
            url = str(value)
            if url and not url.startswith(('http://', 'https://')):
                url = f"https://{url}"
            return {"url": url}
        else:
            return {"rich_text": [{"text": {"content": str(value)[:2000]}}]}
    
    def upload_batch(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Upload multiple business cards to Notion"""
        results = {"success": 0, "failed": 0, "errors": []}
        
        for index, row in df.iterrows():
            # Skip if marked as duplicate and not verified
            if row.get('is_duplicate', False) and not row.get('verified', False):
                results["failed"] += 1
                results["errors"].append(f"Row {index + 1}: Skipped duplicate entry")
                continue
            
            # Create page
            result = self.create_page(row.to_dict())
            
            if result["success"]:
                results["success"] += 1
            else:
                results["failed"] += 1
                results["errors"].append(f"Row {index + 1}: {result['error']}")
        
        return results

def upload_to_notion(df: pd.DataFrame, notion_token: str, notion_database_id: str) -> Dict[str, Any]:
    """
    Upload business card data to Notion database
    
    Args:
        df: DataFrame with business card data
        notion_token: Notion integration token
        notion_database_id: Target database ID
        
    Returns:
        Dictionary with upload results
    """
    if not notion_token or not notion_database_id:
        return {"success": False, "error": "Missing Notion credentials"}
    
    try:
        client = NotionClient(notion_token, notion_database_id)
        
        # Test connection first
        connection_test = client.test_connection()
        if not connection_test["success"]:
            return connection_test
        
        # Upload data
        results = client.upload_batch(df)
        
        return {
            "success": True,
            "results": results,
            "database_title": connection_test.get("database_title", "Unknown")
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}

def validate_notion_credentials(token: str, database_id: str) -> Dict[str, Any]:
    """Validate Notion credentials"""
    if not token or not database_id:
        return {"valid": False, "error": "Missing credentials"}
    
    try:
        client = NotionClient(token, database_id)
        result = client.test_connection()
        
        if result["success"]:
            return {"valid": True, "database_title": result.get("database_title", "Unknown")}
        else:
            return {"valid": False, "error": result["error"]}
    
    except Exception as e:
        return {"valid": False, "error": str(e)}