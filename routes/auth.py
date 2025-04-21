from flask import Blueprint, request, jsonify, current_app, session, make_response, g
from werkzeug.security import check_password_hash
import jwt
import os
import pandas as pd
from datetime import datetime, timedelta
from functools import wraps
from flask_jwt_extended import create_access_token, get_jwt_identity, jwt_required, JWTManager
from werkzeug.utils import secure_filename

from models.user import User, db
from models.portfolio import Portfolio
from models.fst import FST
from models.transactions import Transactions

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%d/%m/%Y").date()
    except Exception:
        return None

# Authentication decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        # Check if token is in headers
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                return jsonify({'message': 'Token is missing!'}), 401
        
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        try:
            # Decode the token
            data = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = User.query.filter_by(id=data['user_id']).first()
            user_portfolio = Portfolio.query.filter_by(portfolio_id=data['portfolio']).first()

            if not current_user:
                return jsonify({'message': 'User not found!'}), 404

        except:
            return jsonify({'message': 'Token is invalid!'}), 401
            
        return f(current_user, user_portfolio, *args, **kwargs)
    
    return decorated

@auth_bp.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    
    # Check if required fields are present
    if not data or not data.get('username') or not data.get('email') or not data.get('password'):
        return jsonify({'message': 'Missing required fields'}), 400
    
    # Check if user already exists
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'message': 'User already exists'}), 409
    
    # Create new user
    new_user = User(
        username=data['username'],
        email=data['email'],
        password=data['password'],
        phonenumber=data.get('phonenumber')
    )

    # Add user to database
    db.session.add(new_user)
    db.session.commit()

    user = User.query.filter_by(email=data['email']).first()
    new_portfolio = Portfolio(
        user_id=user.id,
        real_estate_value=0,
        stock_value=0,
        total_value=0,
        profit_loss=0
    )
    db.session.add(new_portfolio)
    db.session.commit()

    return jsonify({'message': 'User registered successfully'}), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'message': 'Missing username or password'}), 400
    
    user = User.query.filter_by(email=data['email']).first()
    portfolio = Portfolio.query.filter_by(user_id=user.id).first()
    
    if not user:
        return jsonify({'message': 'User not found'}), 404
    
    if user.check_password(data['password']):
        # Update last login time
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        # Generate token
        token = jwt.encode({
            'user_id': user.id,
            'portfolio': portfolio.portfolio_id,
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, current_app.config['SECRET_KEY'], algorithm="HS256")
        
        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': user.to_dict()
        }), 200
    
    return jsonify({'message': 'Invalid password'}), 401

@auth_bp.route('/logout', methods=['POST'])
@token_required
def logout(current_user):
    # In a real application, you might want to blacklist the token
    # For simplicity, we'll just return a success message
    return jsonify({'message': 'Logout successful'}), 200

@auth_bp.route('/me', methods=['GET'])
@token_required
def get_user_profile(current_user):
    return jsonify(current_user.to_dict()), 200

@auth_bp.route('/submit', methods=['POST'])
@token_required
def submit():
    user_id = g.current_user
    
    if not user_id:
        return jsonify({"message": "User ID not found in token"}), 401
    
    data = request.get_json()
    if not data:
        return jsonify({"message": "Missing JSON body"}), 400
    
    required_fields = ["realEstateValue", "stockValue", "totalAssets", "liabilities", 
                      "jamaicaPercentile", "worldPercentile", "jamaicaRank", "worldRank"]
    
    for field in required_fields:
        if field not in data:
            return jsonify({"message": f"Missing field: {field}"}), 400
    
    try:
        # Check if user already has a portfolio entry
        existing_portfolio = Portfolio.query.filter_by(user_id=user_id).first()
        existing_fst = FST.query.filter_by(user_id=user_id).first()
        
        # Calculate profit/loss (total assets minus liabilities)
        profit_loss = data["totalAssets"] - data["liabilities"]
        
        if existing_portfolio:
            # Update existing portfolio record
            existing_portfolio.real_estate_value = data["realEstateValue"]
            existing_portfolio.stock_value = data["stockValue"]
            existing_portfolio.total_value = data["totalAssets"]
            existing_portfolio.profit_loss = profit_loss
        else:
            # Create new portfolio record
            new_portfolio = Portfolio(
                user_id=user_id,
                real_estate_value=data["realEstateValue"],
                stock_value=data["stockValue"],
                total_value=data["totalAssets"],
                profit_loss=profit_loss
            )
            db.session.add(new_portfolio)

        if existing_fst:
            # Update existing FST record
            existing_fst.jam_per = data["jamaicaPercentile"]
            existing_fst.world_per = data["worldPercentile"]
            existing_fst.jam_rank = data["jamaicaRank"]
            existing_fst.world_rank = data["worldRank"]
        else:
            # Create new FST record if it doesn't exist
            new_fst = FST(
                user_id=user_id,
                jam_per=data["jamaicaPercentile"],
                world_per=data["worldPercentile"],
                jam_rank=data["jamaicaRank"],
                world_rank=data["worldRank"]
            )
            db.session.add(new_fst)

        db.session.commit()
        return jsonify({"message": "Successfully Submitted"}), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({"message": "Database error", "error": str(e)}), 500


@auth_bp.route('/finance', methods=["GET", "POST"])
@jwt_required()
def finance():
    current_user_id = get_jwt_identity()  # Get the user ID from JWT token
    
    # Fetch user portfolio data
    portfolio = Portfolio.query.filter_by(user_id=current_user_id).first()
    if not portfolio:
        return jsonify({"message": "Portfolio data not found"}), 404

    # Fetch user financial standing data
    financial_standing = FST.query.filter_by(user_id=current_user_id).first()
    if not financial_standing:
        return jsonify({"message": "Financial standing data not found"}), 404

    # Prepare the data to send back
    dashboard_data = {
        "clientPortfolio": {
            "realEstateValue": portfolio.real_estate_value,
            "stockValue": portfolio.stock_value,
            "totalAssets": portfolio.total_value,
            "liabilities": portfolio.profit_loss,  # Assuming 'profit_loss' refers to liabilities for now
        },
        "financialStanding": {
            "jamaicaPercentile": financial_standing.jam_per,
            "worldPercentile": financial_standing.world_per,
            "jamaicaRank": financial_standing.jam_rank,
            "worldRank": financial_standing.world_rank,
        }
    }

    return jsonify(dashboard_data), 200

@auth_bp.route('/display', methods=["GET"])
@token_required
def display(current_user):
    user_id = current_user.id

    user = User.query.filter_by(id=user_id).first()
    if not user:
        return jsonify({"message": "User Not Found"}), 400

    update_data = {
        "email": user.email, 
        "username": user.username, 
        "phonenumber": user.phonenumber
    }

    return jsonify(update_data), 200

@auth_bp.route('/update', methods=["POST"])
@token_required
def update(current_user):
    current_user_id = current_user.id
    user = User.query.filter_by(id=current_user_id).first()
    if not user: 
        return jsonify({"message": "User Not Found"}), 400

    data = request.get_json()
    password = data.get("currentPassword")
    
    if user.check_password(password):
        newPassword = data.get("newPassword")
        user.username = data.get("username")
        user.email = data.get("email")
        user.phonenumber = data.get("phonenumber")
        user.set_password(newPassword)
    else:
        user.username = data.get("username")
        user.email = data.get("email")
        user.phonenumber = data.get("phonenumber")

    db.session.commit()

    return jsonify({"message": "User Information Updated Successfully"}), 201

@auth_bp.route('/upload', methods=['POST'])
@token_required
def upload_csv(current_user, user_portfolio):
    portfolio_id = user_portfolio.portfolio_id
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        df = pd.read_csv(filepath)
        df['ORDER DATE'] = df['ORDER DATE'].apply(parse_date)

        for _, row in df.iterrows():
            try:
                order = Transactions(
                    portfolio_id=portfolio_id, 
                    order_date=row['ORDER DATE'],
                    equity_order_no=int(row['EQUITY ORDER NO']),
                    status=row['STATUS'],
                    stock_exchange_code=row['STOCK EXCHANGE CODE'],
                    currency=row['CURRENCY'],
                    equity_symbol=row['EQUITY SYMBOL'],
                    order_type=row['ORDER TYPE'],
                    quantity=int(row['QUANTITY']),
                    average_fill_price=float(row['AVERAGE FILL PRICE']),
                    estimated_value=float(row['ESTIMATED VALUE']),
                    time_in_force=row['TIME IN FORCE'],
                    transaction_type=row['TRANSACTION TYPE'],
                    limit_price=float(row['LIMIT PRICE']),
                )
                db.session.add(order)
            except Exception as e:
                print(f"Skipping row: {e}")
        db.session.commit()

        return jsonify({'message': f'{filename} uploaded and processed successfully'}), 200

    return jsonify({'error': 'Invalid file type'}), 400