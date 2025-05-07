from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from models.user import db

class Portfolio(db.Model):
    __tablename__ = 'portfolios'
    
    portfolio_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    real_estate_value = db.Column(db.Integer, nullable=False)
    stock_value = db.Column(db.Integer, nullable=False)
    total_value = db.Column(db.Integer, nullable=False)
    profit_loss = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('portfolio', lazy=True))

    def __init__(self, user_id, real_estate_value, stock_value, total_value, profit_loss):
        self.user_id = user_id
        self.real_estate_value = real_estate_value
        self.stock_value = stock_value
        self.total_value = total_value
        self.profit_loss = profit_loss
        
    def to_dict(self):
        return {
            'portfolio_id': self.portfolio_id,
            'user_id': self.user_id,
            'real_estate_value': self.real_estate_value,
            'stock_value': self.stock_value,
            'total_value': self.total_value,
            'profit_loss': self.profit_loss,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }