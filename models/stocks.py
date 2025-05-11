from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from models.user import db

class Stocks(db.Model):
    __tablename__ = 'stocks'
    
    stock_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    name = db.Column(db.String(80), nullable=True)
    symbol = db.Column(db.String(20), nullable=True)
    close_price = db.Column(db.Float, nullable=True)
    quantity = db.Column(db.Integer, nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('stocks', lazy=True))

    def __init__(self, user_id, name, symbol, close_price, quantity):
        self.user_id = user_id
        self.name = name
        self.symbol = symbol
        self.close_price = close_price
        self.quantity = quantity
        
    def to_dict(self):
        return {
            'user_id': self.user_id,
            'name': self.name,
            'symbol': self.symbol, 
            'close_price': self.close_price, 
            'quantity': self.quantity
        }