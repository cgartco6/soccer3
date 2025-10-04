import requests
import json
from datetime import datetime, timedelta
from config import Config

class RealDataCollector:
    def __init__(self):
        self.odds_api_key = Config.ODDS_API_KEY
        self.sports_api_key = Config.THE_SPORTS_API_KEY
    
    def get_odds_api_data(self, sport='soccer_epl'):
        """Get real odds from Odds API (free tier)"""
        url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
        
        params = {
            'apiKey': self.odds_api_key,
            'regions': 'eu,uk',
            'markets': 'h2h',
            'oddsFormat': 'decimal'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Odds API Error: {response.status_code}")
                return self.get_fallback_data(sport)
        except Exception as e:
            print(f"Odds API Exception: {e}")
            return self.get_fallback_data(sport)
    
    def get_fallback_data(self, sport):
        """Provide fallback data when API fails"""
        # Sample real match data for demonstration
        sample_matches = [
            {
                'id': 'fallback_1',
                'sport_key': sport,
                'sport_title': 'Soccer',
                'home_team': 'Manchester United',
                'away_team': 'Liverpool',
                'commence_time': (datetime.now() + timedelta(hours=2)).isoformat(),
                'bookmakers': [
                    {
                        'key': 'betway',
                        'markets': [
                            {
                                'key': 'h2h',
                                'outcomes': [
                                    {'name': 'Manchester United', 'price': 3.2},
                                    {'name': 'Liverpool', 'price': 2.1},
                                    {'name': 'Draw', 'price': 3.4}
                                ]
                            }
                        ]
                    }
                ]
            },
            {
                'id': 'fallback_2',
                'sport_key': sport,
                'sport_title': 'Soccer',
                'home_team': 'Arsenal',
                'away_team': 'Chelsea',
                'commence_time': (datetime.now() + timedelta(hours=4)).isoformat(),
                'bookmakers': [
                    {
                        'key': 'betway',
                        'markets': [
                            {
                                'key': 'h2h',
                                'outcomes': [
                                    {'name': 'Arsenal', 'price': 2.3},
                                    {'name': 'Chelsea', 'price': 3.0},
                                    {'name': 'Draw', 'price': 3.2}
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
        return sample_matches
    
    def get_live_scores(self):
        """Get live scores from free API"""
        try:
            # Using free football-data.org API
            url = "https://api.football-data.org/v4/matches"
            headers = {'X-Auth-Token': ''}  # Free tier available
            
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                return response.json().get('matches', [])
        except:
            pass
        
        # Return simulated live data
        return self.get_simulated_live_data()
    
    def get_simulated_live_data(self):
        """Simulate live match data for demonstration"""
        matches = [
            {
                'id': 'live_1',
                'homeTeam': {'name': 'Man City'},
                'awayTeam': {'name': 'Tottenham'},
                'score': {'fullTime': {'home': 2, 'away': 1}},
                'status': 'LIVE',
                'minute': 78
            }
        ]
        return matches
    
    def extract_odds(self, match_data, bookmaker='betway'):
        """Extract odds for specific bookmaker"""
        home_odds = away_odds = draw_odds = None
        
        for bookmaker_data in match_data.get('bookmakers', []):
            if bookmaker_data.get('key') == bookmaker:
                for market in bookmaker_data.get('markets', []):
                    if market['key'] == 'h2h':
                        for outcome in market['outcomes']:
                            if outcome['name'] == match_data['home_team']:
                                home_odds = outcome['price']
                            elif outcome['name'] == match_data['away_team']:
                                away_odds = outcome['price']
                            elif outcome['name'] == 'Draw':
                                draw_odds = outcome['price']
        
        return home_odds, away_odds, draw_odds
