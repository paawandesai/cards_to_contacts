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
        # Cache the database properties so we only fetch them once per client instance
        # This allows us to build the property payload dynamically based on what
        # actually exists in the target database, avoiding "property does not exist"
        # errors and type mismatches.
        self.database_properties: Dict[str, Any] = self.get_database_properties()
    
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
        """Map business card data to Notion database properties.

        Only properties that exist in the target database will be included. The
        value is formatted according to the database's property *type* so that
        mismatches (e.g. providing rich_text for a multi_select property) are
        avoided.
        """

        properties: Dict[str, Any] = {}

        # ------------------------------------------------------------------
        # Address parsing – attempt to split the raw address into components
        # like City, State, and Postal Code so they can be mapped to separate
        # columns if the database provides them.
        # ------------------------------------------------------------------

        def _parse_address_components(address: str):
            """Return (city, state, postal_code) if we can parse them, else blanks."""
            if not address:
                return "", "", ""

            # Example pattern: "Springfield, IL 62704" or "Springfield IL 62704"
            # We'll look for a 2-letter state code followed by a 5-digit zip.
            import re

            pattern = re.compile(r"(?P<city>[A-Za-z\s]+)[,\s]+(?P<state>[A-Z]{2})\s+(?P<zip>\d{5}(?:-\d{4})?)")
            match = pattern.search(address)
            if match:
                city = match.group("city").strip()
                state = match.group("state").strip()
                postal = match.group("zip").strip()
                return city, state, postal

            # Fallback: try splitting on commas – last parts may be state/zip.
            parts = [p.strip() for p in address.split(",") if p.strip()]
            if len(parts) >= 2:
                city = parts[-2]
                last_part = parts[-1]
                # Try to split last part into state + zip
                m2 = re.match(r"([A-Z]{2})\s+(\d{5}(?:-\d{4})?)", last_part)
                if m2:
                    return city, m2.group(1), m2.group(2)
            return "", "", ""

        city_val, state_val, postal_val = _parse_address_components(card_data.get("address", ""))

        if not self.database_properties:
            # If we couldn't fetch the schema for some reason fall back to the
            # original static behaviour so we at least attempt to upload.
            return self._legacy_map_card_to_notion_properties(card_data)

        # Mapping of data keys to potential Notion property names.  Feel free to
        # add aliases here – we will pick the first matching property that
        # exists in the database (case-insensitive).
        alias_map: Dict[str, List[str]] = {
            "name": ["Name", "Full Name"],
            "title": ["Title", "Contact title", "Job Title"],
            "company": ["Company", "Company Name", "Organisation", "Organization"],
            "email": ["Email", "E-mail"],
            "phone": ["Phone", "Phone Number", "Mobile"],
            "website": ["Website", "Website URL", "URL"],
            "address": ["Address", "Location"],
            "linkedin": ["LinkedIn", "LinkedIn URL"],
            "additional_notes": ["Notes", "Additional Notes", "Comments"],
            "confidence": ["Confidence", "Score", "Confidence Score"],
            "extracted_date": ["Extracted Date", "Date Extracted", "Date Updated", "Imported"],
            # Address components
            "city": ["City"],
            "state": ["State", "Province"],
            "postal_code": ["Postal Code", "Zip", "Zip Code"]
        }

        # Helper to find a matching property name inside the database schema
        def _find_prop_name(aliases: List[str]) -> str:
            for alias in aliases:
                for db_prop in self.database_properties.keys():
                    if db_prop.lower() == alias.lower():
                        return db_prop  # return the *actual* casing from DB
            return ""

        # Detect the unique title property (there is always exactly one)
        title_prop_name = None
        for db_prop, meta in self.database_properties.items():
            if meta.get("type") == "title":
                title_prop_name = db_prop
                break

        # Iterate through each card field, attempt to map and format.
        for card_field, aliases in alias_map.items():
            if card_field == "extracted_date":
                # This is an internal virtual field that is not present in card_data.
                continue

            # Provide values from parsed components when relevant
            if card_field == "city":
                value = city_val
            elif card_field == "state":
                value = state_val
            elif card_field == "postal_code":
                value = postal_val
            else:
                value = card_data.get(card_field)

            if value is None or (isinstance(value, str) and not value.strip()):
                continue  # Skip empty values

            prop_name = _find_prop_name(aliases)
            # Special handling: if we're mapping the 'name' card field and
            # didn't find an alias match, default to the database's title
            # property so we always populate the primary column.
            if card_field == "name" and not prop_name and title_prop_name:
                prop_name = title_prop_name

            if not prop_name:
                # Property not present in the database – skip it.
                continue

            prop_info = self.database_properties.get(prop_name, {})
            prop_type = prop_info.get("type", "rich_text")

            formatted_value = self._format_notion_property(value, prop_type)
            if formatted_value:
                properties[prop_name] = formatted_value

        # Handle confidence specially if it wasn't already processed (i.e. if the
        # database does not have a matching property we still want to include
        # it as a number in a fallback field)
        if "confidence" in card_data and "confidence" not in alias_map:
            conf_prop_name = _find_prop_name(["Confidence"])
            if conf_prop_name:
                properties[conf_prop_name] = {"number": round(card_data["confidence"], 2)}

        # Always try to capture the extraction/import date if the DB has a date
        # property we can use.
        date_prop_name = _find_prop_name(alias_map["extracted_date"])
        if date_prop_name:
            extracted_value = pd.Timestamp.now().isoformat()
            prop_info = self.database_properties.get(date_prop_name, {})
            prop_type = prop_info.get("type", "date")
            # Re-use the generic formatter so the value matches the property's actual type
            formatted_val = self._format_notion_property(extracted_value, prop_type)
            if formatted_val:
                properties[date_prop_name] = formatted_val

        return properties

    # ------------------------------------------------------------------
    # Legacy (static) mapping retained as a fallback when we cannot obtain
    # the database schema (e.g. network issues).  Original implementation
    # remains unchanged.
    # ------------------------------------------------------------------
    def _legacy_map_card_to_notion_properties(self, card_data: Dict[str, Any]) -> Dict[str, Any]:
        """Previous static mapping logic used before dynamic schema support."""
        properties: Dict[str, Any] = {}

        property_mappings = {
            "name": ("Name", "title", "Contact Name"),
            "title": ("Title", "rich_text", "Contact title"),
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
            if value:
                properties[notion_field] = self._format_notion_property(value, notion_type)

        if "confidence" in card_data:
            properties["Confidence"] = {"number": round(card_data["confidence"], 2)}

        properties["Extracted Date"] = {
            "date": {"start": pd.Timestamp.now().isoformat()}
        }

        return properties
    
    def _format_notion_property(self, value: Any, property_type: str) -> Dict[str, Any]:
        """Format a Python value according to the given Notion property type.

        If the property type is not implemented we gracefully fall back to a
        rich_text representation so that the data is still preserved rather
        than the upload failing entirely.
        """

        try:
            if property_type == "title":
                return {"title": [{"text": {"content": str(value)[:2000]}}]}

            if property_type == "rich_text":
                return {"rich_text": [{"text": {"content": str(value)[:2000]}}]}

            if property_type == "email":
                return {"email": str(value) if "@" in str(value) else None}

            if property_type == "phone_number":
                return {"phone_number": str(value)}

            if property_type == "url":
                url = str(value)
                if url and not url.startswith(("http://", "https://")):
                    url = f"https://{url}"
                return {"url": url}

            if property_type == "multi_select":
                # Split on commas/semicolons/newlines to create separate tags
                if isinstance(value, (list, tuple)):
                    options = [str(v).strip() for v in value if str(v).strip()]
                else:
                    options = [str(v).strip() for v in str(value).replace(";", ",").split(",")]
                return {"multi_select": [{"name": opt[:100]} for opt in options if opt]}

            if property_type == "select":
                return {"select": {"name": str(value)[:100]}}

            if property_type == "number":
                try:
                    num = float(value)
                except (TypeError, ValueError):
                    num = None
                return {"number": num}

            if property_type == "date":
                # Attempt to parse date-like strings; if it fails, default to now
                try:
                    date_val = pd.to_datetime(value).isoformat()
                except Exception:
                    date_val = pd.Timestamp.now().isoformat()
                return {"date": {"start": date_val}}

            # Fallback for unhandled property types
            return {"rich_text": [{"text": {"content": str(value)[:2000]}}]}

        except Exception:
            # In case of any formatting exception fall back to rich_text
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