from flask import Blueprint, request, jsonify
from utils.sentimentAnalysis import NewsSentimentModel
import argparse

sentiment_bp = Blueprint('sentiment', __name__)
MODEL = None

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
        init_sentiment_model()  # Reload model after training
        return jsonify({'message': 'Model trained and saved successfully.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@sentiment_bp.route('/api/analyze-sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.get_json()
        if not MODEL:
            return jsonify({'error': 'Model not loaded'}), 500
        result = MODEL.predict(data['text'])
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@sentiment_bp.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    try:
        data = request.get_json()
        if not MODEL:
            return jsonify({'error': 'Model not loaded'}), 500
        results = [{
            'id': item['id'],
            **MODEL.predict(item['text'])
        } for item in data['items']]
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 400