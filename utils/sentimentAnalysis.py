import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from flask import Flask, request, jsonify
from typing import List, Dict, Any, Optional
import json

# NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


class NewsPreprocessor:
    def __init__(self):
        """Initialize the preprocessor with necessary tools"""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        text = str(text).lower()

        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        text = re.sub(r'@\w+', '', text)

        text = re.sub(r'#', '', text)

        text = re.sub(r'\$\w+', '', text)

        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)

        tokens = word_tokenize(text)

        tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                  if token not in self.stop_words]

        return ''.join(tokens)


class NewsSentimentModel:
    def __init__(self):
        """Initialize the sentiment analysis model"""
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = LogisticRegression(max_iter=1000)
        self.preprocessor = NewsPreprocessor()

    def prepare_data(self, data_path: str):
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            raise ValueError("Unsupported file format. Use .csv or .json")

        if 'text' not in df.columns or 'sentiment' not in df.columns:
            raise ValueError("DataFrame must have 'text' and 'sentiment' columns")

        df['cleaned_text'] = df['text'].apply(self.preprocessor.clean_text)

        return df['cleaned_text'], df['sentiment']

    def train_and_save(self, data_path: str, model_path: str, vectorizer_path: str):
        X, y = self.prepare_data(data_path)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        self.classifier.fit(X_train_vec, y_train)

        y_pred = self.classifier.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"Model Accuracy: {accuracy}")
        print("\nClassification Report:")
        print(report)

        with open(model_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)

        return accuracy


    def predict(self, text: str) -> Dict[str, Any]:
        cleaned_text = self.preprocessor.clean_text(text)

        # Vectorize
        text_vec = self.vectorizer.transform([cleaned_text])

        # Get prediction and confidence
        prediction = self.classifier.predict(text_vec)[0]
        confidence = max(self.classifier.predict_proba(text_vec)[0])

        return {
            'sentiment': int(prediction),
            'confidence': float(confidence)
        }


    @classmethod
    def load_model(cls, model_path: str, vectorizer_path: str):
        """Load a trained model and vectorizer"""
        instance = cls()
        with open(model_path, 'rb') as f:
            instance.classifier = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            instance.vectorizer = pickle.load(f)
        return instance


# Flask application
app = Flask(__name__)

# Load the trained model (global variable)
MODEL = None

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train and save model from given dataset."""
    try:
        data = request.get_json()
        data_path = data['data_path']
        model = NewsSentimentModel()
        model.train_and_save(data_path, 'sentiment_model.pkl', 'vectorizer.pkl')
        return jsonify({'message': 'Model trained and saved successfully.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/analyze-sentiment', methods=['POST'])
def analyze_sentiment():
    """
    Endpoint for sentiment analysis of news content

    Expected JSON input: {"text": "News content here"}
    Returns: {"sentiment": 0 or 1, "confidence": float}
    """
    try:
        data = request.get_json()
        text = data['text']

        if not MODEL:
            return jsonify({'error': 'Model not loaded'}), 500

        result = MODEL.predict(text)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    """
    Endpoint for batch sentiment analysis of multiple news items

    Expected JSON input: {"items": [{"id": "...", "text": "..."}, ...]}
    Returns: {"results": [{"id": "...", "sentiment": 0 or 1, "confidence": float}, ...]}
    """
    try:
        data = request.get_json()
        items = data['items']

        if not MODEL:
            return jsonify({'error': 'Model not loaded'}), 500

        results = []
        for item in items:
            prediction = MODEL.predict(item['text'])
            results.append({
                'id': item['id'],
                **prediction
            })

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


def init_model():
    """Initialize the sentiment analysis model"""
    global MODEL
    try:
        MODEL = NewsSentimentModel.load_model('sentiment_model.pkl', 'vectorizer.pkl')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        MODEL = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the sentiment analysis model.')
    parser.add_argument('data_path', type=str, help='Path to the dataset file')
    parser.add_argument('--model_path', type=str, default='sentiment_model.pkl', help='Path to save the trained model')
    parser.add_argument('--vectorizer_path', type=str, default='vectorizer.pkl', help='Path to save the vectorizer')
    args = parser.parse_args()

    model = NewsSentimentModel()
    model.train_and_save(args.data_path, args.model_path, args.vectorizer_path)