from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import time
import sys

from flask import Blueprint, request, jsonify, current_app, session, make_response, g
from werkzeug.security import check_password_hash
import jwt
import os
import pandas as pd
from datetime import datetime, timedelta, date
from functools import wraps
from flask_jwt_extended import create_access_token, get_jwt_identity, jwt_required, JWTManager
from werkzeug.utils import secure_filename
import psycopg2
from sqlalchemy import create_engine

from models.user import User, db
from models.portfolio import Portfolio
from models.fst import FST
from models.transactions import Transactions
from models.jse_prices import JSE
from models.stocks import Stocks

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'csv'}

def get_filtered_equity_orders(id):
    try:
        # Connect to your PostgreSQL database
        # conn = psycopg2.connect(
        #     host="localhost",        # e.g., "localhost"
        #     dbname="postgres",    # e.g., "equity_db"
        #     user="postgres",
        #     password="admin",
        #     port="5432"         # default is 5432
        # )

        db_uri = current_app.config.get("DATABASE_URI")
        if not db_uri:
            raise ValueError("DATABASE_URI is not configured in app.")

        engine = create_engine(db_uri)

        # Define the SQL query
        query = """
            SELECT 
                "equity_symbol" AS equity_symbol,
                "quantity" AS quantity,
                "transaction_type" AS transaction_type
            FROM transactions
            WHERE "status" = 'FILLED' AND "portfolio_id" = %s;
        """

        print("This is current_app before: ", current_app)
        #engine = db.get_engine(current_app)
        print("This is current_app after: ", current_app)

        # Execute the query and load into a DataFrame
        df = pd.read_sql_query(query, engine, params=(id,))


        stocks = []

        for _, row in df.iterrows():
            print("This is the row: ", row)
            symbol = row['equity_symbol']
            quantity = row['quantity']
            txn_type = row['transaction_type']

            # Check if stock exists
            found = False
            for stock in stocks:
                if stock[0] == symbol:
                    found = True
                    if txn_type == "BUY":
                        stock[1] += quantity
                    elif txn_type == "SELL":
                        stock[1] -= quantity
                        if stock[1] < 0:
                            stock[1] = 0.0
                    break
            
            print("This is the new stocks: ", stocks)
            if not found:
                # Initialize new stock entry
                stocks.append([symbol, quantity if txn_type == "BUY" else 0.0])
        
        for stock in stocks: 
            symbol = stock[0]

            query = """
                    SELECT 
                        "name" AS name,
                        "close_price" AS close_price
                    FROM jse_prices
                    WHERE "symbol" = %s;
                """
            
            df = pd.read_sql_query(query, engine, params=(symbol,))
            print("This is df: ", df)
            #print("This is df: ", df['name'], df['close_price'])

            for _, row in df.iterrows():
                print("This is loop: ", row['name'], row['close_price'])
                stock.append(row["name"])
                stock.append(row["close_price"])


        # conn.close()

        return stocks

    except Exception as e:
        print("Error while connecting to PostgreSQL:", e)
        return None

def get_all_jse_closing_prices(debug=False):
    url = "https://www.jamstockex.com/trading/trade-quotes"
    
    if debug:
        print(f"Opening URL: {url}")

    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")  # Modern headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")  # Set a larger window size
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    # Initialize the driver with improved error handling
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    except Exception as e:
        print(f"Error initializing Chrome driver: {e}")
        print("Make sure you have Chrome installed and ChromeDriver is compatible with your Chrome version.")
        sys.exit(1)
    
    try:
        # Navigate to the page
        driver.get(url)
        if debug:
            print("Page loaded. Waiting for table...")
        
        # Try to find any table on the page
        try:
            tables = WebDriverWait(driver, 0.5).until(
                EC.presence_of_all_elements_located((By.TAG_NAME, "table"))
            )
            table = tables[2]
        except TimeoutException:
            print("Couldn't find tables")

        # Extract data from the table
        rows = table.find_elements(By.TAG_NAME, "tr")
        
        # Proceed with data extraction
        data = []
        for row in rows[1:]:  # Skip the header row
            try:
                names = row.find_elements(By.TAG_NAME, "a")
                close_price = row.find_elements(By.TAG_NAME, "td")
                
                # Debug the column content
                if debug and names and close_price:
                    for name in names:
                        #for j in close_price:
                            company_name = name.get_attribute('title')
                            company_abb = name.get_attribute('innerHTML').strip().replace("\n", "")
                            company_price = close_price[3].get_attribute('innerHTML').strip().replace("\n", "")
                            comp_info = [company_name, company_abb, float(company_price)]
                    name = comp_info[0]
                    symbol = comp_info[1]
                    close = comp_info[2]
                    data.append((name, symbol, close))

            except Exception as e:
                if debug:
                    print(f"Error processing row: {e}")
                continue
        
        # Create DataFrame from collected data
        df = pd.DataFrame(data, columns=["Name", "Symbol", "Close"])
        df["trade_date"] = date.today().isoformat()
        
        return df
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame(columns=["name", "Symbol", "Close", "trade_date"])
    
    finally:
        # Always close the driver
        try:
            driver.quit()
            if debug:
                print("Browser closed")
        except Exception as e:
            print(f"Error closing browser: {e}")
            
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
                return jsonify({'message': 'Token is misplaced!'}), 401
                print("This is token: ", token)
        
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

        # df = get_all_jse_closing_prices(debug=True)

        # if not df.empty:
        #     print("This is df: ", df)

        #     JSE.query.delete()
        #     db.session.commit()

        #     for _, row in df.iterrows():
        #         try:
        #             stocks = JSE(
        #                 name = row['Name'], 
        #                 symbol = row['Symbol'], 
        #                 close_price = row['Close']
        #             )
        #             db.session.add(stocks)
        #         except Exception as e:
        #             print(f"Skipping row: {e}")
        #     db.session.commit()
        # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        # filename = secure_filename(file.filename)
        # filepath = os.path.join(UPLOAD_FOLDER, filename)
        # file.save(filepath)

        # df = pd.read_csv(filepath)
        # df['ORDER DATE'] = df['ORDER DATE'].apply(parse_date)

        # for _, row in df.iterrows():
        #     try:
        #         order = Transactions(
        #             portfolio_id=portfolio_id, 
        #             order_date=row['ORDER DATE'],
        #             equity_order_no=int(row['EQUITY ORDER NO']),
        #             status=row['STATUS'],
        #             stock_exchange_code=row['STOCK EXCHANGE CODE'],
        #             currency=row['CURRENCY'],
        #             equity_symbol=row['EQUITY SYMBOL'],
        #             order_type=row['ORDER TYPE'],
        #             quantity=int(row['QUANTITY']),
        #             average_fill_price=float(row['AVERAGE FILL PRICE']),
        #             estimated_value=float(row['ESTIMATED VALUE']),
        #             time_in_force=row['TIME IN FORCE'],
        #             transaction_type=row['TRANSACTION TYPE'],
        #             limit_price=float(row['LIMIT PRICE']),
        #         )
        #         db.session.add(order)
        #     except Exception as e:
        #         print(f"Skipping row: {e}")
        # db.session.commit()
        
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
def submit(current_user, user_portfolio):
    user_id = current_user.id
    
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
@token_required
def finance(current_user, user_portfolio):
    current_user_id = current_user.id  # Get the user ID from JWT token

    assets = []
    net_worth = 0.0
    # Fetch user portfolio data
    stocks = Stocks.query.filter_by(user_id=current_user_id).all()
    if not stocks:
        return jsonify({"message": "Stocks data not found"}), 404

    print("This is stocks: ", stocks)

    for stock in stocks:
        amt = stock.close_price * stock.quantity
        net_worth += amt
        print("This is stock: ", stock.name, amt)
        val = {"symbol": stock.symbol, "name": stock.name, "price": stock.close_price, "shares": stock.quantity, "value": amt}
        assets.append(val)

    print("This is assets: ", assets)
    print("This is net_worth: ", net_worth)

    portfolio = Portfolio.query.filter_by(user_id=current_user_id).first()
    print("This is portfolio stock value: ", portfolio.stock_value)
    #print("This is their sum:", net_worth + portfolio.stock_value)

    netWorth = net_worth + portfolio.stock_value

    portfolio.stock_value = net_worth
    db.session.commit()

    final_val = {"netWorth": net_worth}

    dashboard_data = {
        "assets": assets,
        "netWorth": final_val
    }
    


    # Fetch user financial standing data
    # financial_standing = FST.query.filter_by(user_id=current_user_id).first()
    # if not financial_standing:
    #     return jsonify({"message": "Financial standing data not found"}), 404

    #Prepare the data to send back

    return jsonify(dashboard_data), 200

@auth_bp.route('/display', methods=["GET"])
@token_required
def display(current_user, user_portfolio):
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
def update(current_user, user_portfolio):
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
    #portfolio_id = user_portfolio.portfolio_id

    portfolio = Portfolio.query.filter_by(user_id=current_user.id).first()
    if not portfolio:
        return jsonify({'error': 'Portfolio not found'}), 404
    
    portfolio_id = portfolio.portfolio_id
    print("This is portfolio id: ", portfolio_id)
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

        df = get_filtered_equity_orders(portfolio_id)
        print("This is df: ", df)

        for row in df:
            try:
                stock = Stocks.query.filter_by(symbol=row[0], user_id=current_user.id).first()
                if not stock:
                    order = Stocks(
                        user_id=current_user.id, 
                        name=row[2],
                        symbol=row[0], 
                        close_price=row[3], 
                        quantity=int(row[1]),
                    )
                    db.session.add(order)
                    db.session.commit()
                else:
                    stock.quantity += int(row[1])
                    stock.close_price = row[3]
                    stock.name = row[2]
                    db.session.commit()

            except Exception as e:
                print(f"Skipping row: {e}")

        return jsonify({'message': f'{filename} uploaded and processed successfully'}), 200

    return jsonify({'error': 'Invalid file type'}), 400

@auth_bp.route('/jse-prices', methods=['GET'])
@token_required
def get_jse_prices(current_user, user_portfolio):
    """
    Retrieve all JSE price data from the database
    
    Returns:
        JSON response with list of all JSE prices
    """
    try:
        # Query all records from JSE table
        jse_prices = JSE.query.all()
        
        if not jse_prices:
            return jsonify({"message": "No JSE price data found"}), 404
        
        # Format the data for the response
        prices_data = []
        for price in jse_prices:
            prices_data.append({
                "id": price.stock_id,
                "name": price.name,
                "symbol": price.symbol,
                "close_price": price.close_price
            })
        
        return jsonify({"jse_prices": prices_data}), 200
    
    except Exception as e:
        # Log the error for debugging
        print(f"Error retrieving JSE prices: {e}")
        return jsonify({"message": "Error retrieving JSE prices", "error": str(e)}), 500