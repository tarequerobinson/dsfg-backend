from flask import Blueprint, jsonify
from functools import wraps

from routes.auth import token_required

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