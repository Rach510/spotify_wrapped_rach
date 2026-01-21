import streamlit as st
import requests
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from frontend.frontend_config import API_BASE_URL, APP_TITLE, APP_ICON, PAGE_LAYOUT

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Inter:wght@300;500;700&display=swap');


* {
    font-family: 'Inter', sans-serif;
}

body {
    background: linear-gradient(180deg, #021024, #052659);
    color: #C1E8FF;
}

.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 0.5rem;
    color: #C1E8FF;
    font-family: 'Playfair Display', serif;
    letter-spacing: 1px;
}

.sub-header {
    font-size: 1.1rem;
    text-align: center;
    color: #7DA0CA;
    margin-bottom: 2rem;
    font-family: 'Playfair Display', serif;
}


.feature-box {
    background: rgba(84, 131, 179, 0.15); /* #5483B3 */
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 25px;
    margin: 15px 0;
    border-left: 6px solid #7DA0CA;
}

div[data-testid="metric-container"] {
    background: rgba(84, 131, 179, 0.15);
    border-radius: 16px;
    padding: 10px;
    border: 1px solid rgba(193,232,255,0.2);
}

.stButton > button {
    background: linear-gradient(90deg, #5483B3, #7DA0CA);
    color: #021024;
    border-radius: 12px;
    border: none;
    padding: 0.5rem 1.2rem;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)




# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_api_health():
    """Check if Flask API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return True, data
        return False, None
    except Exception as e:
        return False, str(e)


def upload_csv_file(uploaded_file):
    """Upload CSV file to Flask API"""
    try:
        files = {'file': uploaded_file}
        response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=30)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get('error', 'Unknown error')
    except Exception as e:
        return False, str(e)


def get_mood_distribution():
    """Get mood distribution from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/mood-distribution", timeout=10)
        if response.status_code == 200:
            return True, response.json()
        return False, response.json().get('error', 'Unknown error')
    except Exception as e:
        return False, str(e)


def get_stats():
    """Get user stats from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        if response.status_code == 200:
            return True, response.json()
        return False, response.json().get('error', 'Unknown error')
    except Exception as e:
        return False, str(e)


def get_top_artists(n=10):
    """Get top artists from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/top-artists?n={n}", timeout=10)
        if response.status_code == 200:
            return True, response.json()
        return False, response.json().get('error', 'Unknown error')
    except Exception as e:
        return False, str(e)


# ============================================================================
# MAIN APP
# ============================================================================

# Title
st.markdown('<div class="main-header">🎵 Rach's Spotify Wrapped 🎵</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">A behavioral analysis of your 2025 music taste</div>', unsafe_allow_html=True)

# Check API health
api_healthy, health_data = check_api_health()

if not api_healthy:
    st.error("⚠️ Cannot connect to Flask API!")
    st.error(f"Error: {health_data}")
    st.info("Please make sure the Flask API is running on http://localhost:5000")
    st.code("python backend/spotify_api_dynamic.py", language="bash")
    st.stop()

# Display API status
with st.expander("ℹ️ API Status", expanded=False):
    if health_data:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("API Status", "✅ Connected")
            st.metric("Recommender Status", 
                     "✅ Loaded" if health_data.get('recommender_loaded') else "❌ Not Loaded")
        with col2:
            st.metric("User Data", 
                     f"{health_data.get('user_data_rows', 0)} tracks" if health_data.get('user_data_loaded') else "Not uploaded")
            st.metric("Recommendation Database", 
                     f"{health_data.get('recommender_tracks', 0)} tracks")

st.divider()

# ============================================================================
# SECTION 1: RECOMMENDATIONS INFO
# ============================================================================

st.header("Recs we think you'd love")

st.markdown("""
<div class='feature-box'>
    <h3>🎵 How It Works</h3>
    <ol>
        <li><strong>Rate 10 Songs:</strong> We'll show you 10 random songs from our vast music database</li>
        <li><strong>Give Your Opinion:</strong> Rate each song from 1-5 stars based on your preference</li>
        <li><strong>Get Recommendations:</strong> Our AI analyzes your ratings and finds songs you'll love!</li>
    </ol>
    <br>
    <p style='text-align: center;'>
        👉 <strong>Go to the "Recommendations" page to get started!</strong> 👈
    </p>
</div>
""", unsafe_allow_html=True)

if health_data and not health_data.get('recommender_loaded'):
    st.warning("⚠️ Recommender model not loaded. Please run:")
    st.code("python backend/train_recommender.py", language="bash")

st.divider()

# ============================================================================
# SECTION 2: FILE UPLOAD (For Personalized Analysis)
# ============================================================================

st.header("📤 Upload Your Spotify Data")
st.markdown("Upload your Spotify playlist CSV to get personalized mood and listening analysis")

uploaded_file = st.file_uploader(
    "Choose your Spotify CSV file",
    type=['csv'],
    help="Export your Spotify playlist as CSV and upload it here"
)

if uploaded_file is not None:
    with st.spinner("Uploading and analyzing your data..."):
        success, result = upload_csv_file(uploaded_file)
    
    if success:
        st.success(f"✅ Successfully uploaded {result['rows']} tracks!")
        
        # Store upload status in session state
        st.session_state['data_uploaded'] = True
        st.session_state['upload_info'] = result
        
        with st.expander("📊 Upload Summary", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Tracks", result['rows'])
            with col2:
                if result['preview'].get('total_duration_ms'):
                    hours = result['preview']['total_duration_ms'] / (1000 * 60 * 60)
                    st.metric("Total Hours", f"{hours:.1f}")
            with col3:
                st.metric("Columns", len(result['columns']))
    else:
        st.error(f"❌ Upload failed: {result}")

st.divider()

# ============================================================================
# SECTION 3: MOOD ANALYSIS (Requires Upload)
# ============================================================================

if st.session_state.get('data_uploaded', False):
    st.header("🎧 Your Music Mood Profile")
    
    with st.spinner("Analyzing your mood distribution..."):
        success, mood_data = get_mood_distribution()
    
    if success:
        mood_dist = mood_data['mood_distribution']
        
        # Get top mood
        top_mood = max(mood_dist.items(), key=lambda x: x[1]['count'])[0]
        
        st.subheader("🎵 Your Top Mood")
        st.success(f"Your most common vibe is... **{top_mood}** 🎉")
        
        # Create mood chart
        st.subheader("📊 Mood Distribution")
        
        mood_df = pd.DataFrame([
            {'Mood': mood, 'Percentage': data['percentage']}
            for mood, data in mood_dist.items()
        ])
        
        st.bar_chart(mood_df.set_index('Mood'))
        
        # Mood insights
        st.markdown("### 🔥 Mood Insights")
        
        if top_mood == "Energetic":
            st.write("💪 Such an Adrenaline and Dopamine Addict")
        elif top_mood == "Chill":
            st.write("🌙 You're just a chill person like that")
        elif top_mood == "Happy":
            st.write("✨ You enjoy upbeat, feel-good music – Let's catch those feels!")
        elif top_mood == "Sad":
            st.write("🎭 You lean towards emotional tracks – i feel ya")
        
        # Additional stats
        success_stats, stats_data = get_stats()
        if success_stats:
            st.subheader("📈 Your Listening Stats")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Tracks", stats_data['total_tracks'])
                if 'unique_artists' in stats_data:
                    st.metric("Unique Artists", stats_data['unique_artists'])
            
            with col2:
                if 'total_duration' in stats_data:
                    st.metric("Total Hours", stats_data['total_duration']['hours'])
                if 'popularity' in stats_data:
                    st.metric("Avg Popularity", stats_data['popularity']['average'])
            
            with col3:
                if 'explicit' in stats_data:
                    st.metric("Explicit Content", f"{stats_data['explicit']['percentage']}%")
        
        # Top Artists
        success_artists, artists_data = get_top_artists(10)
        if success_artists:
            st.subheader("🎤These people live rent free in your head!")
            
            artists_df = pd.DataFrame(artists_data['top_artists'])
            
            for _, artist in artists_df.iterrows():
                st.markdown(f"**{artist['rank']}. {artist['artist']}** - {artist['track_count']} tracks ({artist['percentage']}%)")
    
    else:
        st.error(f"Failed to get mood data: {mood_data}")
    
    st.divider()

else:
    st.info("📤 Upload your Spotify CSV above to see your personalized mood analysis!")
    st.info("🎯 Or go to the Recommendations page to discover new music without uploading!")
    st.divider()

st.divider()

st.markdown("### 🎬 Final verdict")
st.write("If my life was a movie, it would be written by my top artists and directed by my emotional instability.")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🎵 Powered by Flask API + Streamlit + Machine Learning 🎵</p>
    <p>Made with ❤️ for music lovers</p>
</div>
""", unsafe_allow_html=True)
