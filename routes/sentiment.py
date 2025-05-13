from flask import Blueprint, request, jsonify
import json
from datetime import datetime
import logging

from utils.geminiSentimentModel import analyze_with_gemini
from utils.sentimentAnalysis import NewsSentimentModel

sentiment_bp = Blueprint('sentiment', __name__)
MODEL = None

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sentiment.log')
    ]
)
def init_sentiment_model():
    global MODEL
    try:
        MODEL = NewsSentimentModel.load_model('sentiment_model.pkl', 'vectorizer.pkl')
        print("Sentiment model loaded successfully")
    except Exception as e:
        print(f"Error loading sentiment model: {e}")
        MODEL = None


@sentiment_bp.route('/api/train', methods=['POST'])
def train_model():
    try:
        data = request.get_json()
        model = NewsSentimentModel()
        model.train_and_save(data['data_path'], 'sentiment_model.pkl', 'vectorizer.pkl')
        init_sentiment_model()
        return jsonify({'message': 'Model trained and saved successfully.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@sentiment_bp.route('/api/analyze-sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text parameter'}), 400

        if not MODEL:
            logging.warning("Local model not loaded, falling back to Gemini API")
            result = analyze_with_gemini(data['text'])
            return jsonify(result)

        result = MODEL.predict(data['text'])
        return jsonify(result)
    except Exception as e:
        logging.error(f"Analysis error: {e}")
        return jsonify({'error': str(e)}), 400


@sentiment_bp.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    try:
        data = request.get_json()
        results = []

        for item in data['items']:
            """results.append({
                'id': item['id'],
                **MODEL.predict(item['text'])
            })"""
            analysis = analyze_with_gemini(item['text'])
            results.append({
                'id': item['id'],
                'sentiment': analysis['sentiment'],
                'confidence': analysis['confidence']
            })
        logging.debug(f"{results}")

        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
