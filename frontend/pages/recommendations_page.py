"""
Spotify Wrapped - Rating-Based Recommendation Engine
Rate 10 songs and get personalized recommendations
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.api_client import APIClient

# Page config
st.set_page_config(
    page_title="Recommendations",
    page_icon="🎯",
    layout="wide"
)

# Apply Spotify theme
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #191414 0%, #1a1a1a 100%);
    }
    h1, h2, h3 { color: #1DB954 !important; }
    p, li, label { color: #FFFFFF !important; }
    .stButton>button {
        background-color: #1DB954;
        color: white;
        border-radius: 30px;
        padding: 10px 30px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { 
        background-color: #1ed760;
        transform: scale(1.05);
    }
    .song-card {
        background-color: #282828;
        border-radius: 15px;
        padding: 30px;
        margin: 20px 0;
        border-left: 5px solid #1DB954;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .rec-card {
        background-color: #282828;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #1DB954;
        transition: all 0.3s ease;
    }
    .rec-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(29, 185, 84, 0.3);
    }
    .star-rating {
        font-size: 40px;
        cursor: pointer;
        user-select: none;
    }
    .star-filled {
        color: #FFD700;
    }
    .star-empty {
        color: #666;
    }
    .progress-bar {
        background-color: #282828;
        border-radius: 10px;
        padding: 15px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize API client
api = APIClient()

# Initialize session state
if 'rating_phase' not in st.session_state:
    st.session_state.rating_phase = True
if 'current_song_index' not in st.session_state:
    st.session_state.current_song_index = 0
if 'songs_to_rate' not in st.session_state:
    st.session_state.songs_to_rate = None
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {}
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'current_rating' not in st.session_state:
    st.session_state.current_rating = 0


def reset_session():
    """Reset all session state variables"""
    st.session_state.rating_phase = True
    st.session_state.current_song_index = 0
    st.session_state.songs_to_rate = None
    st.session_state.user_ratings = {}
    st.session_state.recommendations = None
    st.session_state.current_rating = 0


def load_songs_to_rate():
    """Load 10 random songs from API"""
    with st.spinner("Loading songs for you to rate..."):
        data = api.start_rating_session()
        if data and 'songs' in data:
            st.session_state.songs_to_rate = data['songs']
            st.session_state.current_song_index = 0
            st.session_state.user_ratings = {}
            return True
        else:
            st.error("Failed to load songs. Please try again.")
            return False


def render_star_rating(song_id, current_rating):
    """Render interactive star rating"""
    st.markdown("### ⭐ Rate this song (1-5 stars)")
    
    cols = st.columns(5)
    rating = current_rating
    
    for i, col in enumerate(cols, 1):
        with col:
            if st.button(
                "⭐" if i <= current_rating else "☆",
                key=f"star_{song_id}_{i}",
                help=f"Rate {i} star{'s' if i > 1 else ''}"
            ):
                rating = i
                st.session_state.current_rating = i
    
    return rating


def render_rating_phase():
    """Render the rating interface"""
    st.markdown("<h1 style='text-align: center;'>🎯 Rate Songs & Get Recommendations</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px; color: #b3b3b3;'>Rate 10 songs to discover your personalized playlist</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Load songs if not loaded
    if st.session_state.songs_to_rate is None:
        if not load_songs_to_rate():
            return
    
    songs = st.session_state.songs_to_rate
    current_idx = st.session_state.current_song_index
    
    # Progress bar
    progress = (current_idx + 1) / len(songs)
    st.progress(progress)
    st.markdown(f"""
    <div class='progress-bar'>
        <h3 style='text-align: center; margin: 0;'>Song {current_idx + 1} of {len(songs)}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Current song
    current_song = songs[current_idx]
    
    # Display song card
    st.markdown(f"""
    <div class='song-card'>
        <h2 style='margin-bottom: 10px;'>{current_song['track_name']}</h2>
        <p style='font-size: 18px; color: #b3b3b3; margin: 5px 0;'>
            <strong>Artist:</strong> {current_song['artists']}
        </p>
        <p style='font-size: 16px; color: #b3b3b3; margin: 5px 0;'>
            <strong>Genre:</strong> {current_song.get('track_genre', 'N/A')}
        </p>
    """, unsafe_allow_html=True)
    
    # Show additional info if available
    col1, col2 = st.columns(2)
    with col1:
        if 'year' in current_song:
            st.markdown(f"<p style='color: #b3b3b3;'><strong>Year:</strong> {current_song['year']}</p>", unsafe_allow_html=True)
    with col2:
        if 'popularity' in current_song:
            st.markdown(f"<p style='color: #b3b3b3;'><strong>Popularity:</strong> {current_song['popularity']}/100</p>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Star rating
    rating = render_star_rating(current_song['id'], st.session_state.current_rating)
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if current_idx > 0:
            if st.button("⬅️ Previous", key="prev_btn"):
                # Save current rating
                if st.session_state.current_rating > 0:
                    st.session_state.user_ratings[current_idx] = {
                        'df_index': current_song['df_index'],
                        'rating': st.session_state.current_rating
                    }
                
                st.session_state.current_song_index -= 1
                # Load previous rating if exists
                prev_idx = st.session_state.current_song_index
                if prev_idx in st.session_state.user_ratings:
                    st.session_state.current_rating = st.session_state.user_ratings[prev_idx]['rating']
                else:
                    st.session_state.current_rating = 0
                st.rerun()
    
    with col3:
        is_last_song = current_idx == len(songs) - 1
        button_text = "Get Recommendations 🎵" if is_last_song else "Next Song ➡️"
        
        # Check if current song is rated
        can_proceed = st.session_state.current_rating > 0
        
        if not can_proceed:
            st.warning("⭐ Please rate this song before continuing")
        
        if st.button(button_text, key="next_btn", disabled=not can_proceed):
            # Save current rating
            st.session_state.user_ratings[current_idx] = {
                'df_index': current_song['df_index'],
                'rating': st.session_state.current_rating
            }
            
            if is_last_song:
                # Submit ratings and get recommendations
                submit_ratings()
            else:
                # Move to next song
                st.session_state.current_song_index += 1
                # Load next rating if exists
                next_idx = st.session_state.current_song_index
                if next_idx in st.session_state.user_ratings:
                    st.session_state.current_rating = st.session_state.user_ratings[next_idx]['rating']
                else:
                    st.session_state.current_rating = 0
                st.rerun()
    
    # Show rated songs summary
    if st.session_state.user_ratings:
        with st.expander(f"✅ Rated Songs ({len(st.session_state.user_ratings)}/10)"):
            for idx, rating_data in st.session_state.user_ratings.items():
                song = songs[idx]
                stars = "⭐" * rating_data['rating']
                st.markdown(f"**{song['track_name']}** by {song['artists']} - {stars}")


def submit_ratings():
    """Submit ratings and get recommendations"""
    with st.spinner("🎵 Generating your personalized recommendations..."):
        # Convert ratings dict to list format required by API
        ratings_list = [
            st.session_state.user_ratings[i] 
            for i in range(len(st.session_state.songs_to_rate))
        ]
        
        # Call API
        result = api.submit_ratings_and_recommend(ratings_list, top_k=10)
        
        if result:
            st.session_state.recommendations = result
            st.session_state.rating_phase = False
            st.rerun()
        else:
            st.error("Failed to get recommendations. Please try again.")


def render_recommendations_phase():
    """Render the recommendations results"""
    st.markdown("<h1 style='text-align: center;'>🎵 Your Personalized Recommendations</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px; color: #b3b3b3;'>Based on your ratings</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    if not st.session_state.recommendations:
        st.error("No recommendations available")
        return
    
    recs = st.session_state.recommendations
    
    # Show summary
    st.success(f"✅ Found {recs['count']} personalized recommendations!")
    st.info(f"📊 {recs['based_on']} • {recs['source']}")
    
    st.markdown("### 🎧 Your Recommended Tracks")
    
    # Display recommendations
    for i, rec in enumerate(recs['recommendations'], 1):
        st.markdown(f"""
        <div class='rec-card'>
            <h3 style='margin: 0 0 10px 0;'>#{i} {rec['track_name']}</h3>
            <p style='margin: 5px 0; font-size: 16px;'><strong>Artist:</strong> {rec['artists']}</p>
            <p style='margin: 5px 0; font-size: 14px; color: #b3b3b3;'><strong>Genre:</strong> {rec.get('track_genre', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional info
        col1, col2, col3 = st.columns(3)
        with col1:
            if 'year' in rec:
                st.metric("Year", rec['year'])
        with col2:
            if 'popularity' in rec:
                st.metric("Popularity", f"{rec['popularity']}/100")
        with col3:
            if 'similarity_score' in rec and rec['similarity_score']:
                match_pct = rec['similarity_score'] * 100
                st.metric("Match", f"{match_pct:.1f}%")
        
        st.markdown("---")
    
    # Start over button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Rate New Songs", key="start_over"):
            reset_session()
            st.rerun()


def main():
    """Main app logic"""
    
    # Check if rating phase or recommendations phase
    if st.session_state.rating_phase:
        render_rating_phase()
    else:
        render_recommendations_phase()


if __name__ == "__main__":
    main()
