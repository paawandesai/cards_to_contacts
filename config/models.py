"""
GPT Model configurations and pricing for AI Business Card Reader
"""

MODEL_OPTIONS = {
    "gpt-4o-mini": {
        "name": "GPT-4o Mini (Most Affordable)",
        "input_cost": 0.00015,  # per 1K tokens
        "output_cost": 0.0006,  # per 1K tokens
        "vision_cost": 0.00764,  # per image
        "description": "Fast and cost-effective for basic business card extraction"
    },
    "gpt-4o": {
        "name": "GPT-4o (Best Performance)",
        "input_cost": 0.005,    # per 1K tokens
        "output_cost": 0.015,   # per 1K tokens
        "vision_cost": 0.01528, # per image
        "description": "Premium model with highest accuracy for complex business cards"
    }
}

def get_model_info(model_key):
    """Get model information by key"""
    return MODEL_OPTIONS.get(model_key, MODEL_OPTIONS["gpt-4o-mini"])

def calculate_estimated_cost(model_key, num_images):
    """Calculate estimated cost for processing images"""
    model_info = get_model_info(model_key)
    return num_images * model_info["vision_cost"]

def format_cost(cost):
    """Format cost for display"""
    return f"${cost:.4f}"