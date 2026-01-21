"""
Session State Manager for Spotify Wrapped
Centralized session state management
"""

import streamlit as st

class SessionManager:
    """Manages Streamlit session state"""
    
    @staticmethod
    def init_session_state():
        """Initialize all session state variables"""
        defaults = {
            # Upload state
            'data_uploaded': False,
            'upload_info': None,
            
            # Wrapped page state
            'feature_index': 0,
            'wrapped_completed': False,
            
            # Recommendations state
            'selected_track': None,
            'recommendations': None,
            'track_names': None,
            'random_songs': None,
            
            # Cache for API responses
            'cached_stats': None,
            'cached_top_artists': None,
            'cached_top_tracks': None,
            'cached_mood_dist': None,
            'cached_listening_age': None,
            'cached_playlist_age': None,
            'cached_popularity_dist': None,
            'cached_temporal': None,
            
            # UI state
            'show_welcome': True,
            'current_page': 'home'
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def reset_upload_data():
        """Reset upload-related data"""
        st.session_state.data_uploaded = False
        st.session_state.upload_info = None
        st.session_state.feature_index = 0
        st.session_state.wrapped_completed = False
        SessionManager.clear_cache()
    
    @staticmethod
    def clear_cache():
        """Clear all cached API responses"""
        cache_keys = [
            'cached_stats',
            'cached_top_artists',
            'cached_top_tracks',
            'cached_mood_dist',
            'cached_listening_age',
            'cached_playlist_age',
            'cached_popularity_dist',
            'cached_temporal'
        ]
        
        for key in cache_keys:
            if key in st.session_state:
                st.session_state[key] = None
    
    @staticmethod
    def get_cached_or_fetch(cache_key, fetch_function):
        """Get cached data or fetch from API"""
        if st.session_state.get(cache_key) is None:
            st.session_state[cache_key] = fetch_function()
        return st.session_state[cache_key]
    
    @staticmethod
    def mark_wrapped_complete():
        """Mark wrapped experience as complete"""
        st.session_state.wrapped_completed = True
    
    @staticmethod
    def reset_wrapped():
        """Reset wrapped to beginning"""
        st.session_state.feature_index = 0
        st.session_state.wrapped_completed = False
    
    @staticmethod
    def next_feature():
        """Move to next feature"""
        st.session_state.feature_index += 1
    
    @staticmethod
    def previous_feature():
        """Move to previous feature"""
        if st.session_state.feature_index > 0:
            st.session_state.feature_index -= 1
    
    @staticmethod
    def get_upload_status():
        """Get upload status information"""
        return {
            'uploaded': st.session_state.data_uploaded,
            'info': st.session_state.upload_info
        }
