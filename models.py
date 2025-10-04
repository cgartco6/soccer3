from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Match(db.Model):
    __tablename__ = 'matches'
    
    id = db.Column(db.Integer, primary_key=True)
    match_id = db.Column(db.String(100), unique=True, nullable=False)
    sport_key = db.Column(db.String(50), nullable=False)
    home_team = db.Column(db.String(100), nullable=False)
    away_team = db.Column(db.String(100), nullable=False)
    commence_time = db.Column(db.DateTime, nullable=False)
    
    # Compact odds storage
    home_odds = db.Column(db.Float)
    away_odds = db.Column(db.Float)
    draw_odds = db.Column(db.Float)
    bookmaker = db.Column(db.String(50))  # 'betway' or 'hollywoodbets'
    
    # AI Predictions (compressed)
    home_win_prob = db.Column(db.Float)
    away_win_prob = db.Column(db.Float)
    draw_prob = db.Column(db.Float)
    predicted_winner = db.Column(db.String(10))  # 'home', 'away', 'draw'
    confidence = db.Column(db.Float)
    value_bet = db.Column(db.Boolean, default=False)
    value_side = db.Column(db.String(10))
    edge = db.Column(db.Float)  # Value bet edge percentage
    
    # Live data
    is_live = db.Column(db.Boolean, default=False)
    home_score = db.Column(db.Integer, default=0)
    away_score = db.Column(db.Integer, default=0)
    status = db.Column(db.String(20))
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
