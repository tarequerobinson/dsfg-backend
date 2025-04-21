from flask_sqlalchemy import SQLAlchemy
from models.portfolio import db

class Transactions(db.Model):
    __tablename__ = 'transactions'

    trans_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    portfolio_id = db.Column(db.Integer, db.ForeignKey('portfolios.portfolio_id'), nullable=False)
    order_date = db.Column(db.Date)
    equity_order_no = db.Column(db.Integer, unique=True, nullable=False)
    status = db.Column(db.String(50))
    stock_exchange_code = db.Column(db.String(10))
    currency = db.Column(db.String(10))
    equity_symbol = db.Column(db.String(20))
    order_type = db.Column(db.String(20))
    quantity = db.Column(db.Integer)
    average_fill_price = db.Column(db.Float)
    estimated_value = db.Column(db.Float)
    time_in_force = db.Column(db.String(10))
    transaction_type = db.Column(db.String(10))
    limit_price = db.Column(db.Float)
    user = db.relationship('Portfolio', backref=db.backref('transactions', lazy=True))


    def __init__(self, portfolio_id, order_date, equity_order_no, status, stock_exchange_code, 
                 currency, equity_symbol, order_type, quantity, average_fill_price, estimated_value, 
                 time_in_force, transaction_type, limit_price):
        self.portfolio_id = portfolio_id
        self.order_date = order_date
        self.equity_order_no = equity_order_no
        self.status = status
        self.stock_exchange_code = stock_exchange_code
        self.currency = currency
        self.equity_symbol = equity_symbol
        self.order_type = order_type
        self.quantity = quantity
        self.average_fill_price = average_fill_price
        self.estimated_value = estimated_value
        self.time_in_force = time_in_force
        self.transaction_type = transaction_type
        self.limit_price = limit_price

    def to_dict(self):
        return {
            "portfolio_id": self.portfolio_id,
            "order_date": self.order_date.isoformat() if self.order_date else None,
            "equity_order_no": self.equity_order_no,
            "status": self.status,
            "stock_exchange_code": self.stock_exchange_code,
            "currency": self.currency,
            "equity_symbol": self.equity_symbol,
            "order_type": self.order_type,
            "quantity": self.quantity,
            "average_fill_price": self.average_fill_price,
            "estimated_value": self.estimated_value,
            "time_in_force": self.time_in_force,
            "transaction_type": self.transaction_type,
            "limit_price": self.limit_price
        }
