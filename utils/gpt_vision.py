"""
GPT Vision API integration for business card extraction
"""

import base64
import json
import re
import streamlit as st
from openai import OpenAI
from PIL import Image
import io
from typing import Dict, List, Any

EXTRACTION_PROMPT = """
IMPORTANT: Respond with ONLY valid JSON. No extra text, explanations, or markdown formatting.

Analyze this image and identify all business cards present. Return this exact JSON structure:

{
  "cards": [
    {
      "card_number": 1,
      "confidence": 0.95,
      "extracted_data": {
        "name": "",
        "title": "",
        "company": "",
        "email": "",
        "phone": "",
        "website": "",
        "address": "",
        "linkedin": "",
        "additional_notes": ""
      }
    }
  ]
}

Rules:
- Return ONLY the JSON object above, nothing else
- Do not wrap in code blocks or markdown
- Do not add explanatory text before or after
- Be thorough and extract ALL text visible on each card
- For email, phone, and websites, extract all instances found
- Provide confidence scores between 0-1 for each card based on text clarity
- If multiple cards are in the image, return data for each separately
- If no business cards are found, return empty cards array
- Extract exactly what you see - don't infer or guess missing information
- For phone numbers, preserve the original format
- For addresses, include full address if available
- Include any social media handles or additional contact information in additional_notes
"""

def extract_json_from_response(content: str) -> Dict[str, Any]:
    """
    Robustly extract JSON from GPT response handling various formats
    
    Args:
        content: Raw response content from GPT
        
    Returns:
        Extracted JSON data or error information
    """
    if not content:
        return {"error": "Empty response", "cards": []}
    
    # List of JSON extraction strategies
    strategies = [
        # Strategy 1: Direct JSON parsing
        lambda text: json.loads(text.strip()),
        
        # Strategy 2: Remove markdown code blocks
        lambda text: json.loads(re.sub(r'```(?:json)?\n?(.*?)\n?```', r'\1', text, flags=re.DOTALL).strip()),
        
        # Strategy 3: Find JSON object with regex
        lambda text: json.loads(re.search(r'\{.*\}', text, re.DOTALL).group(0)),
        
        # Strategy 4: Extract JSON between first { and last }
        lambda text: json.loads(text[text.find('{'):text.rfind('}') + 1]),
        
        # Strategy 5: Remove common prefixes and suffixes
        lambda text: json.loads(re.sub(r'^.*?(\{.*\}).*?$', r'\1', text, flags=re.DOTALL)),
        
        # Strategy 6: Find JSON starting with "cards" (improved pattern)
        lambda text: json.loads(re.search(r'\{[^{]*"cards"[^{]*\[.*?\]\s*\}', text, re.DOTALL).group(0)),
    ]
    
    # Try each strategy
    for i, strategy in enumerate(strategies):
        try:
            result = strategy(content)
            
            # Validate the extracted JSON has required structure
            if isinstance(result, dict) and "cards" in result:
                # Ensure cards is a list
                if not isinstance(result["cards"], list):
                    result["cards"] = []
                
                # Validate each card structure
                validated_cards = []
                for card in result["cards"]:
                    if isinstance(card, dict):
                        # Ensure required fields exist
                        validated_card = {
                            "card_number": card.get("card_number", len(validated_cards) + 1),
                            "confidence": min(max(card.get("confidence", 0.5), 0.0), 1.0),
                            "extracted_data": card.get("extracted_data", {})
                        }
                        
                        # Ensure extracted_data has all required fields
                        required_fields = ["name", "title", "company", "email", "phone", "website", "address", "linkedin", "additional_notes"]
                        for field in required_fields:
                            if field not in validated_card["extracted_data"]:
                                validated_card["extracted_data"][field] = ""
                        
                        validated_cards.append(validated_card)
                
                result["cards"] = validated_cards
                return result
        
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            # Strategy failed, try next one
            continue
    
    # If all strategies failed, try to extract individual fields manually
    extracted_data = extract_fallback_data(content)
    return {
        "cards": [extracted_data] if extracted_data else [],
        "error": "Failed to parse as JSON, used fallback extraction",
        "raw_response": content[:500]  # Truncate for display
    }

def extract_fallback_data(content: str) -> Dict[str, Any]:
    """
    Fallback extraction when JSON parsing fails completely
    Extract data using regex patterns
    """
    # Common patterns for business card information
    patterns = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'[\+]?[1-9]?[0-9]{0,3}[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
        "website": r'(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?',
        "linkedin": r'linkedin\.com/in/[a-zA-Z0-9-]+',
    }
    
    extracted = {
        "name": "",
        "title": "",
        "company": "",
        "email": "",
        "phone": "",
        "website": "",
        "address": "",
        "linkedin": "",
        "additional_notes": content[:200]  # Store raw content as notes
    }
    
    # Extract using patterns
    for field, pattern in patterns.items():
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            extracted[field] = matches[0]
    
    # If we found any useful data, return it
    if any(extracted[field] for field in ["email", "phone", "website"]):
        return {
            "card_number": 1,
            "confidence": 0.3,  # Low confidence for fallback extraction
            "extracted_data": extracted
        }
    
    return None

def encode_image(image_file) -> str:
    """Convert uploaded image to base64 string"""
    try:
        # Open image with PIL
        image = Image.open(image_file)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr = img_byte_arr.getvalue()
        
        # Encode to base64
        return base64.b64encode(img_byte_arr).decode('utf-8')
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def extract_business_cards(image_file, model: str, api_key: str) -> Dict[str, Any]:
    """
    Extract business card information using GPT Vision
    
    Args:
        image_file: Uploaded image file
        model: GPT model to use
        api_key: OpenAI API key
        
    Returns:
        Dictionary containing extracted cards and usage info
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Encode image
        base64_image = encode_image(image_file)
        if not base64_image:
            return {"error": "Failed to process image"}
        
        # Make API call
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": EXTRACTION_PROMPT
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        # Parse response
        content = response.choices[0].message.content
        
        # Use robust JSON extraction
        result = extract_json_from_response(content)
        
        # Validate and clean extracted data
        if "cards" in result:
            result["cards"] = validate_extracted_data(result["cards"])
        
        # Add usage information
        result["usage"] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        # Show appropriate message based on extraction result
        if "error" in result:
            if "fallback extraction" in result["error"]:
                st.warning("⚠️ Response format was not valid JSON, used fallback extraction")
            else:
                st.warning(f"⚠️ {result['error']}")
        
        return result
    
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg:
            st.error("❌ Invalid or expired OpenAI API key")
        elif "rate limit" in error_msg.lower():
            st.error("❌ OpenAI API rate limit exceeded. Please try again later.")
        elif "content policy" in error_msg.lower():
            st.error("❌ Image content may violate OpenAI's usage policies")
        else:
            st.error(f"❌ API call failed: {error_msg}")
        
        return {
            "error": f"API call failed: {error_msg}",
            "cards": []
        }

def validate_extracted_data(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate and clean extracted business card data
    
    Args:
        cards: List of extracted business card data
        
    Returns:
        List of validated and cleaned card data
    """
    validated_cards = []
    
    for i, card in enumerate(cards):
        if not isinstance(card, dict):
            continue
            
        # Ensure required structure
        validated_card = {
            "card_number": card.get("card_number", i + 1),
            "confidence": min(max(float(card.get("confidence", 0.5)), 0.0), 1.0),
            "extracted_data": {}
        }
        
        # Extract and validate data fields
        raw_data = card.get("extracted_data", {})
        if not isinstance(raw_data, dict):
            raw_data = {}
        
        # Required fields with validation
        field_validators = {
            "name": lambda x: str(x).strip()[:100] if x else "",
            "title": lambda x: str(x).strip()[:100] if x else "",
            "company": lambda x: str(x).strip()[:100] if x else "",
            "email": lambda x: str(x).strip()[:100] if x and "@" in str(x) else "",
            "phone": lambda x: str(x).strip()[:50] if x else "",
            "website": lambda x: str(x).strip()[:200] if x else "",
            "address": lambda x: str(x).strip()[:200] if x else "",
            "linkedin": lambda x: str(x).strip()[:100] if x else "",
            "additional_notes": lambda x: str(x).strip()[:500] if x else ""
        }
        
        # Apply validation to each field
        for field, validator in field_validators.items():
            try:
                validated_card["extracted_data"][field] = validator(raw_data.get(field, ""))
            except (ValueError, TypeError):
                validated_card["extracted_data"][field] = ""
        
        # Only add cards with at least some useful information
        has_useful_data = any([
            validated_card["extracted_data"]["name"],
            validated_card["extracted_data"]["email"],
            validated_card["extracted_data"]["phone"],
            validated_card["extracted_data"]["company"]
        ])
        
        if has_useful_data:
            validated_cards.append(validated_card)
    
    return validated_cards

def calculate_actual_cost(usage: Dict[str, int], model: str) -> float:
    """Calculate actual cost based on token usage"""
    from config.models import get_model_info
    
    model_info = get_model_info(model)
    
    prompt_cost = (usage.get("prompt_tokens", 0) / 1000) * model_info["input_cost"]
    completion_cost = (usage.get("completion_tokens", 0) / 1000) * model_info["output_cost"]
    
    return prompt_cost + completion_cost

def validate_api_key(api_key: str) -> bool:
    """Validate OpenAI API key"""
    try:
        client = OpenAI(api_key=api_key)
        # Make a simple API call to test the key
        client.models.list()
        return True
    except Exception:
        return False