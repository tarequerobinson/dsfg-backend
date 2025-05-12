import json

import google.generativeai as genai
import os

from functools import lru_cache
from dotenv import load_dotenv
from flask import logging
import logging
from datetime import datetime

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sentiment.log')
    ]
)
def analyze_with_gemini(text: str) -> dict:
    """
    Analyzes text sentiment using Gemini API
    Returns: {sentiment: -1, 0, 1, confidence: float}
    """
    prompt = f"""Analyze this financial news sentiment. Respond ONLY with a JSON object containing:
    - "sentiment" (integer): -1 (negative), 0 (neutral), or 1 (positive)
    - "confidence" (float): 0.0 to 1.0
    
    Text: {text}
    
    Example response: {{"sentiment": 1, "confidence": 0.95}}"""

    try:
        logging.debug(f"Analyzing text: {text[:50]}...")
        response = model.generate_content(prompt)
        logging.debug(f"Full API response: {response}")

        try:
            parsed = eval(response.text)
        except Exception as parse_error:
            logging.error(f"Failed to parse response: {response.text}")
            logging.error(f"Parse error: {parse_error}")
            return {"sentiment": 0, "confidence": 0.0}

        return parsed
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        return {"sentiment": 0, "confidence": 0.0}


