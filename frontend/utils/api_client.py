"""
API Client for Flask Backend Communication
Handles all requests to the Spotify Wrapped API
"""

import requests
import streamlit as st

class APIClient:
    """Client for communicating with Flask API"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
    
    def _make_request(self, method, endpoint, **kwargs):
        """Make HTTP request with error handling"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.request(method, url, timeout=10, **kwargs)
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = response.json().get('error', 'Request failed')
                st.error(f"API Error: {error_msg}")
                return None
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Please ensure Flask backend is running on port 5000.")
            return None
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. Please try again.")
            return None
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {str(e)}")
            return None
    
    # ============================================================================
    # USER DATA ANALYSIS ENDPOINTS
    # ============================================================================
    
    def get_stats(self):
        """Get overall statistics"""
        return self._make_request('GET', '/stats')
    
    def get_top_artists(self, n=10):
        """Get top artists"""
        return self._make_request('GET', f'/top-artists?n={n}')
    
    def get_top_tracks(self, n=10):
        """Get top tracks"""
        return self._make_request('GET', f'/top-tracks?n={n}')
    
    def get_mood_distribution(self):
        """Get mood distribution"""
        return self._make_request('GET', '/mood-distribution')
    
    def get_listening_age(self):
        """Get listening age"""
        return self._make_request('GET', '/listening-age')
    
    def get_playlist_age(self):
        """Get playlist age"""
        return self._make_request('GET', '/playlist-age')
    
    def get_popularity_distribution(self):
        """Get popularity distribution"""
        return self._make_request('GET', '/popularity-distribution')
    
    def get_explicit_analysis(self):
        """Get explicit content analysis"""
        return self._make_request('GET', '/explicit-analysis')
    
    def get_temporal_analysis(self):
        """Get temporal analysis (songs added over time)"""
        return self._make_request('GET', '/temporal-analysis')
    
    # ============================================================================
    # RATING-BASED RECOMMENDATION ENDPOINTS
    # ============================================================================
    
    def start_rating_session(self):
        """Get 10 random songs for rating"""
        return self._make_request('GET', '/start-rating-session')
    
    def submit_ratings_and_recommend(self, ratings, top_k=10):
        """
        Submit user ratings and get personalized recommendations
        
        Parameters:
        -----------
        ratings : list
            List of rating objects: [{'df_index': 123, 'rating': 4}, ...]
        top_k : int
            Number of recommendations to return
        
        Returns:
        --------
        dict or None
            Recommendations data or None if failed
        """
        data = {
            'ratings': ratings,
            'top_k': top_k
        }
        return self._make_request('POST', '/submit-ratings-and-recommend', json=data)
    
    # ============================================================================
    # UTILITY
    # ============================================================================
    
    def health_check(self):
        """Check API health"""
        return self._make_request('GET', '/health')
