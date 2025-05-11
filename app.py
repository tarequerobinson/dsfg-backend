from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from datetime import timedelta ,datetime
import pandas as pd
from werkzeug.utils import secure_filename
from routes.sentiment import sentiment_bp, init_sentiment_model

# Load environment variables
load_dotenv()

# Import routes
from routes.auth import auth_bp
from routes.main import main_bp


def create_app(test_config=None):
    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    
    # Enable CORS
    CORS(app)
    
    # Load configuration
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev'),
        DATABASE_URI=os.environ.get('DATABASE_URI', 'sqlite:///app.db'),
        DEBUG=os.environ.get('DEBUG', 'False') == 'True',
        JWT_SECRET_KEY=os.environ.get('JWT_SECRET_KEY', os.environ.get('SECRET_KEY', 'dev')),
        JWT_ACCESS_TOKEN_EXPIRES=timedelta(seconds=int(os.environ.get('JWT_ACCESS_TOKEN_EXPIRES', 3600))),
        UPLOAD_FOLDER=os.environ.get('UPLOAD_FOLDER', 'dev')
    )

    if test_config is None:
        # Load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # Load the test config if passed in
        app.config.from_mapping(test_config)

    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass


    # Initialize JWT
    jwt = JWTManager(app)
    init_sentiment_model()
    
    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(sentiment_bp)

    # Initialize database
    from utils.db import init_db
    init_db(app)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
