from flask import Blueprint, jsonify, request
from functools import wraps
import os
import jwt
import pandas as pd
from datetime import datetime
from werkzeug.utils import secure_filename

from routes.auth import token_required
from models.transactions import Transactions
from models.user import User
from models.portfolio import Portfolio
from utils.db import db

main_bp = Blueprint('main', __name__, url_prefix='/api')

@main_bp.route('/', methods=['GET'])
def index():
    return jsonify({
        'message': 'Welcome to the DSFG API',
        'status': 'online'
    }), 200

@main_bp.route('/protected', methods=['GET'])
@token_required
def protected(current_user):
    return jsonify({
        'message': 'This is a protected route',
        'user': current_user.to_dict()
    }), 200