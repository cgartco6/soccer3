import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database - Using SQLite to save space
    SQLALCHEMY_DATABASE_URI = 'sqlite:///sports_betting.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Free API Keys (No cost)
    ODDS_API_KEY = os.getenv('ODDS_API_KEY', 'demo_key')  # Free tier: 500 req/month
    THE_SPORTS_API_KEY = os.getenv('THE_SPORTS_API_KEY', 'demo_key')  # Free tier
    
    # Update intervals to save API calls
    UPDATE_INTERVAL = 600  # 10 minutes
    LIVE_UPDATE_INTERVAL = 300  # 5 minutes
    
    # Sports to monitor (saves space)
    MONITORED_SPORTS = ['soccer', 'basketball']
    MONITORED_LEAGUES = ['soccer_epl', 'soccer_laliga', 'soccer_bundesliga', 'basketball_nba']
