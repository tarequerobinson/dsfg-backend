from flask_sqlalchemy import SQLAlchemy
from models.portfolio import db

class JSE(db.Model):
    __tablename__ = 'jse_prices'
    
    stock_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(80), nullable=True)
    symbol = db.Column(db.String(20), nullable=True)
    close_price = db.Column(db.Float, nullable=True)

    def __init__(self, name, symbol, close_price):
        self.name = name
        self.symbol = symbol
        self.close_price = close_price