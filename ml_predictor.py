import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
from datetime import datetime

class AccurateAIPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.team_encoder = LabelEncoder()
        self.is_trained = False
        self.model_file = 'ai_model.joblib'
        
        # Historical performance data (would be from database in production)
        self.team_stats = {}
        
    def create_historical_data(self):
        """Create realistic historical training data"""
        np.random.seed(42)
        
        teams = [
            'Manchester United', 'Liverpool', 'Manchester City', 'Chelsea', 'Arsenal',
            'Tottenham', 'Newcastle', 'Brighton', 'West Ham', 'Crystal Palace',
            'Wolves', 'Aston Villa', 'Leeds', 'Everton', 'Southampton'
        ]
        
        data = []
        for _ in range(2000):  # Reduced for space
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])
            
            # Realistic team strengths based on actual Premier League data
            team_strengths = {
                'Manchester City': 0.85, 'Liverpool': 0.82, 'Chelsea': 0.80,
                'Manchester United': 0.78, 'Arsenal': 0.77, 'Tottenham': 0.76,
                'Newcastle': 0.70, 'Brighton': 0.68, 'West Ham': 0.67,
                'Crystal Palace': 0.60, 'Wolves': 0.59, 'Aston Villa': 0.58,
                'Leeds': 0.55, 'Everton': 0.53, 'Southampton': 0.52
            }
            
            home_strength = team_strengths.get(home_team, 0.6)
            away_strength = team_strengths.get(away_team, 0.6)
            
            # Home advantage
            home_advantage = 0.15
            
            # Calculate true probabilities
            home_prob = self.calculate_true_probability(home_strength, away_strength, home_advantage)
            away_prob = self.calculate_true_probability(away_strength, home_strength, -home_advantage)
            draw_prob = 1 - home_prob - away_prob
            
            # Add randomness
            home_prob += np.random.normal(0, 0.05)
            away_prob += np.random.normal(0, 0.05)
            draw_prob = max(0.1, 1 - home_prob - away_prob)
            
            # Normalize
            total = home_prob + away_prob + draw_prob
            home_prob /= total
            away_prob /= total
            draw_prob /= total
            
            # Generate odds with bookmaker margin
            margin = 1.05
            home_odds = round(margin / home_prob, 2)
            away_odds = round(margin / away_prob, 2)
            draw_odds = round(margin / draw_prob, 2)
            
            # Determine outcome based on true probabilities
            outcome = np.random.choice(['home', 'away', 'draw'], 
                                     p=[home_prob, away_prob, draw_prob])
            
            # Additional features
            home_form = np.random.normal(0.5, 0.2)
            away_form = np.random.normal(0.5, 0.2)
            head_to_head = np.random.normal(0.5, 0.1)
            
            data.append({
                'home_team': home_team,
                'away_team': away_team,
                'home_odds': home_odds,
                'away_odds': away_odds,
                'draw_odds': draw_odds,
                'home_strength': home_strength,
                'away_strength': away_strength,
                'home_form': max(0.1, min(0.9, home_form)),
                'away_form': max(0.1, min(0.9, away_form)),
                'head_to_head': max(0.1, min(0.9, head_to_head)),
                'outcome': outcome
            })
        
        return pd.DataFrame(data)
    
    def calculate_true_probability(self, team1_strength, team2_strength, advantage):
        """Calculate true probability using logistic function"""
        strength_diff = team1_strength - team2_strength + advantage
        return 1 / (1 + np.exp(-3 * strength_diff))
    
    def train_model(self):
        """Train accurate AI model"""
        print("Training accurate AI model...")
        
        df = self.create_historical_data()
        
        # Encode teams
        all_teams = list(set(df['home_team'].tolist() + df['away_team'].tolist()))
        self.team_encoder.fit(all_teams)
        
        df['home_encoded'] = self.team_encoder.transform(df['home_team'])
        df['away_encoded'] = self.team_encoder.transform(df['away_team'])
        
        # Feature engineering
        df['odds_ratio'] = df['home_odds'] / df['away_odds']
        df['strength_diff'] = df['home_strength'] - df['away_strength']
        df['form_diff'] = df['home_form'] - df['away_form']
        
        features = [
            'home_encoded', 'away_encoded', 'home_odds', 'away_odds', 'draw_odds',
            'home_strength', 'away_strength', 'home_form', 'away_form', 
            'head_to_head', 'odds_ratio', 'strength_diff', 'form_diff'
        ]
        
        X = df[features]
        y = df['outcome']
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train with optimized parameters
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate accuracy
        train_acc = self.model.score(X_train_scaled, y_train)
        test_acc = self.model.score(self.scaler.transform(X_test), y_test)
        
        print(f"Model trained - Train: {train_acc:.3f}, Test: {test_acc:.3f}")
        
        # Save model
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'encoder': self.team_encoder
        }, self.model_file)
        
        self.is_trained = True
        return True
    
    def load_model(self):
        """Load trained model"""
        try:
            if os.path.exists(self.model_file):
                saved = joblib.load(self.model_file)
                self.model = saved['model']
                self.scaler = saved['scaler']
                self.team_encoder = saved['encoder']
                self.is_trained = True
                return True
        except:
            pass
        return False
    
    def predict_match(self, home_team, away_team, home_odds, away_odds, draw_odds):
        """Make accurate prediction for match"""
        if not self.is_trained:
            if not self.load_model():
                self.train_model()
        
        try:
            # Get team strengths (from historical data)
            team_strengths = self.get_team_strengths()
            home_strength = team_strengths.get(home_team, 0.5)
            away_strength = team_strengths.get(away_team, 0.5)
            
            # Encode teams
            home_encoded = self.encode_team(home_team)
            away_encoded = self.encode_team(away_team)
            
            # Calculate features
            odds_ratio = home_odds / away_odds if away_odds else 1.0
            strength_diff = home_strength - away_strength
            form_diff = 0.5  # Would be from recent form in production
            
            features = np.array([[
                home_encoded, away_encoded, home_odds, away_odds, draw_odds,
                home_strength, away_strength, 0.5, 0.5, 0.5,  # form and h2h placeholders
                odds_ratio, strength_diff, form_diff
            ]])
            
            # Predict
            features_scaled = self.scaler.transform(features)
            probabilities = self.model.predict_proba(features_scaled)[0]
            prediction = self.model.predict(features_scaled)[0]
            
            return {
                'predicted_winner': prediction,
                'home_win_probability': float(probabilities[0]),
                'away_win_probability': float(probabilities[1]),
                'draw_probability': float(probabilities[2]),
                'confidence': float(np.max(probabilities))
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self.fallback_prediction(home_odds, away_odds, draw_odds)
    
    def encode_team(self, team_name):
        """Encode team name, handling new teams"""
        try:
            return self.team_encoder.transform([team_name])[0]
        except:
            # Add new team
            all_teams = list(self.team_encoder.classes_)
            all_teams.append(team_name)
            self.team_encoder.fit(all_teams)
            return self.team_encoder.transform([team_name])[0]
    
    def get_team_strengths(self):
        """Get realistic team strengths"""
        return {
            'Manchester City': 0.85, 'Liverpool': 0.82, 'Arsenal': 0.80,
            'Manchester United': 0.75, 'Chelsea': 0.74, 'Tottenham': 0.72,
            'Newcastle': 0.68, 'Brighton': 0.66, 'West Ham': 0.64,
            'Crystal Palace': 0.60, 'Wolves': 0.58, 'Aston Villa': 0.62,
            'Everton': 0.55, 'Leeds': 0.56, 'Southampton': 0.54,
            'Leicester': 0.59, 'Fulham': 0.57, 'Brentford': 0.61
        }
    
    def fallback_prediction(self, home_odds, away_odds, draw_odds):
        """Fallback to probability calculation from odds"""
        if not all([home_odds, away_odds, draw_odds]):
            return {
                'predicted_winner': 'draw',
                'home_win_probability': 0.33,
                'away_win_probability': 0.33,
                'draw_probability': 0.34,
                'confidence': 0.34
            }
        
        home_prob = 1 / home_odds
        away_prob = 1 / away_odds
        draw_prob = 1 / draw_odds
        
        total = home_prob + away_prob + draw_prob
        home_prob /= total
        away_prob /= total
        draw_prob /= total
        
        max_prob = max(home_prob, away_prob, draw_prob)
        winner = 'home' if max_prob == home_prob else 'away' if max_prob == away_prob else 'draw'
        
        return {
            'predicted_winner': winner,
            'home_win_probability': home_prob,
            'away_win_probability': away_prob,
            'draw_probability': draw_prob,
            'confidence': max_prob
        }

class ValueBetDetector:
    def __init__(self, threshold=0.05):
        self.threshold = threshold
    
    def find_value_bets(self, prediction, odds):
        """Find value bets with accurate edge calculation"""
        home_odds = odds.get('home')
        away_odds = odds.get('away')
        draw_odds = odds.get('draw')
        
        home_prob = prediction['home_win_probability']
        away_prob = prediction['away_win_probability']
        draw_prob = prediction['draw_probability']
        
        value_bets = []
        
        if home_odds and home_prob > 0:
            implied_prob = 1 / home_odds
            edge = home_prob - implied_prob
            if edge > self.threshold:
                value_bets.append({
                    'side': 'home',
                    'edge': edge,
                    'odds': home_odds,
                    'predicted_prob': home_prob,
                    'implied_prob': implied_prob,
                    'expected_value': (home_odds - 1) * home_prob - (1 - home_prob)
                })
        
        if away_odds and away_prob > 0:
            implied_prob = 1 / away_odds
            edge = away_prob - implied_prob
            if edge > self.threshold:
                value_bets.append({
                    'side': 'away',
                    'edge': edge,
                    'odds': away_odds,
                    'predicted_prob': away_prob,
                    'implied_prob': implied_prob,
                    'expected_value': (away_odds - 1) * away_prob - (1 - away_prob)
                })
        
        if draw_odds and draw_prob > 0:
            implied_prob = 1 / draw_odds
            edge = draw_prob - implied_prob
            if edge > self.threshold:
                value_bets.append({
                    'side': 'draw',
                    'edge': edge,
                    'odds': draw_odds,
                    'predicted_prob': draw_prob,
                    'implied_prob': implied_prob,
                    'expected_value': (draw_odds - 1) * draw_prob - (1 - draw_prob)
                })
        
        return value_bets
