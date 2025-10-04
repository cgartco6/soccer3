from flask import Flask, render_template, jsonify, request
from config import Config
from models import db, Match
from data_collector import RealDataCollector
from ml_predictor import AccurateAIPredictor, ValueBetDetector
from datetime import datetime, timedelta
import schedule
import time
import threading

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

# Initialize components
data_collector = RealDataCollector()
ai_predictor = AccurateAIPredictor()
value_detector = ValueBetDetector(threshold=0.03)  # 3% edge

def init_db():
    with app.app_context():
        db.create_all()
        print("Database initialized")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/matches')
def get_matches():
    """Get matches with AI predictions"""
    try:
        sport = request.args.get('sport', 'soccer')
        show_live = request.args.get('live', 'false').lower() == 'true'
        show_value = request.args.get('value', 'false').lower() == 'true'
        
        query = Match.query
        
        if sport != 'all':
            query = query.filter(Match.sport_key.contains(sport))
        
        if show_live:
            query = query.filter(Match.is_live == True)
        
        if show_value:
            query = query.filter(Match.value_bet == True)
        
        matches = query.order_by(Match.commence_time).limit(50).all()  # Limit for performance
        
        return jsonify({
            'success': True,
            'matches': [match.to_dict() for match in matches]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/matches/update')
def update_matches():
    """Update matches from real APIs"""
    try:
        updated = fetch_real_matches()
        return jsonify({
            'success': True,
            'updated': updated,
            'message': f'Updated {updated} matches'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/predict/custom', methods=['POST'])
def predict_custom():
    """Predict custom match"""
    try:
        data = request.get_json()
        
        home_team = data.get('home_team', '').strip()
        away_team = data.get('away_team', '').strip()
        home_odds = float(data.get('home_odds', 2.0))
        away_odds = float(data.get('away_odds', 2.0))
        draw_odds = float(data.get('draw_odds', 3.0))
        
        if not home_team or not away_team:
            return jsonify({'success': False, 'error': 'Team names required'})
        
        # Get AI prediction
        prediction = ai_predictor.predict_match(
            home_team, away_team, home_odds, away_odds, draw_odds
        )
        
        # Find value bets
        value_bets = value_detector.find_value_bets(prediction, {
            'home': home_odds, 'away': away_odds, 'draw': draw_odds
        })
        
        best_value = max(value_bets, key=lambda x: x['edge']) if value_bets else None
        
        result = {
            'success': True,
            'prediction': prediction,
            'value_bets': value_bets,
            'best_value_bet': best_value,
            'input': {
                'home_team': home_team,
                'away_team': away_team,
                'home_odds': home_odds,
                'away_odds': away_odds,
                'draw_odds': draw_odds
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/live/update')
def update_live():
    """Update live scores"""
    try:
        updated = update_live_scores()
        return jsonify({
            'success': True,
            'updated': updated,
            'message': f'Updated {updated} live matches'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def fetch_real_matches():
    """Fetch real matches from APIs"""
    updated = 0
    
    for league in Config.MONITORED_LEAGUES:
        try:
            matches_data = data_collector.get_odds_api_data(league)
            
            for match_data in matches_data:
                if process_match_data(match_data, 'betway'):
                    updated += 1
                    
        except Exception as e:
            print(f"Error processing {league}: {e}")
    
    return updated

def process_match_data(match_data, bookmaker):
    """Process match data and save to database"""
    try:
        match_id = match_data.get('id')
        home_team = match_data.get('home_team')
        away_team = match_data.get('away_team')
        commence_time = datetime.fromisoformat(
            match_data['commence_time'].replace('Z', '+00:00')
        )
        
        # Extract odds
        home_odds, away_odds, draw_odds = data_collector.extract_odds(match_data, bookmaker)
        
        if not all([home_odds, away_odds, draw_odds]):
            return False
        
        # Check if match exists
        existing = Match.query.filter_by(match_id=match_id).first()
        
        if existing:
            match = existing
        else:
            match = Match(match_id=match_id)
        
        # Update match data
        match.sport_key = match_data.get('sport_key', 'soccer')
        match.home_team = home_team
        match.away_team = away_team
        match.commence_time = commence_time
        match.home_odds = home_odds
        match.away_odds = away_odds
        match.draw_odds = draw_odds
        match.bookmaker = bookmaker
        
        # Get AI prediction
        prediction = ai_predictor.predict_match(
            home_team, away_team, home_odds, away_odds, draw_odds
        )
        
        match.home_win_prob = prediction['home_win_probability']
        match.away_win_prob = prediction['away_win_probability']
        match.draw_prob = prediction['draw_probability']
        match.predicted_winner = prediction['predicted_winner']
        match.confidence = prediction['confidence']
        
        # Check for value bets
        value_bets = value_detector.find_value_bets(prediction, {
            'home': home_odds, 'away': away_odds, 'draw': draw_odds
        })
        
        match.value_bet = len(value_bets) > 0
        match.value_side = value_bets[0]['side'] if value_bets else None
        match.edge = value_bets[0]['edge'] if value_bets else 0
        
        # Check if live
        time_diff = datetime.utcnow().replace(tzinfo=commence_time.tzinfo) - commence_time
        match.is_live = timedelta(0) <= time_diff <= timedelta(hours=3)
        
        if not existing:
            db.session.add(match)
        
        db.session.commit()
        return True
        
    except Exception as e:
        print(f"Error processing match: {e}")
        db.session.rollback()
        return False

def update_live_scores():
    """Update live scores"""
    updated = 0
    try:
        live_data = data_collector.get_live_scores()
        
        for live_match in live_data:
            home_team = live_match.get('homeTeam', {}).get('name')
            away_team = live_match.get('awayTeam', {}).get('name')
            
            if home_team and away_team:
                match = Match.query.filter(
                    (Match.home_team.contains(home_team)) | 
                    (Match.away_team.contains(away_team))
                ).first()
                
                if match:
                    score = live_match.get('score', {})
                    match.home_score = score.get('fullTime', {}).get('home', 0)
                    match.away_score = score.get('fullTime', {}).get('away', 0)
                    match.status = live_match.get('status', 'LIVE')
                    match.is_live = True
                    db.session.commit()
                    updated += 1
                    
    except Exception as e:
        print(f"Live update error: {e}")
    
    return updated

def background_updates():
    """Background data updates"""
    schedule.every(10).minutes.do(fetch_real_matches)
    schedule.every(2).minutes.do(update_live_scores)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == '__main__':
    init_db()
    
    # Start background updates
    bg_thread = threading.Thread(target=background_updates, daemon=True)
    bg_thread.start()
    
    # Initial data load
    fetch_real_matches()
    
    print("🚀 Sports Betting AI Platform Started!")
    print("📊 Access at: http://localhost:5000")
    print("💎 AI predictions and value bets ready")
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
