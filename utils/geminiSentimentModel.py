import google.generativeai as genai
import os

from charset_normalizer.md import lru_cache
from dotenv import load_dotenv
from flask import logging
import logging

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"),
                transport="rest",
                client_options={"api_endpoint": "generativelanguage.googleapis.com/v1"})
model = genai.GenerativeModel('gemini-1.0-pro')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to terminal
        logging.FileHandler('sentiment.log')  # Optional: Log to file
    ]
)
@lru_cache(maxsize=1000)
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
        logging.debug(f"Analyzing text: {text[:50]}...")  # Log first 50 chars
        response = model.generate_content(prompt)
        logging.debug(f"Full API response: {response}")

        parsed = eval(response.text)
        logging.info(f"Analysis result: {parsed} | Processing time: {datetime.now() - start_time}")

        return parsed
    except Exception as e:
        print(f"Gemini API error: {e}")
        return {"sentiment": 0, "confidence": 0.0}