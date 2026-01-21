"""
Frontend Configuration and Styling
Centralized configuration for the Streamlit app
"""

# API Configuration
API_BASE_URL = "http://localhost:5000"
# App metadata
APP_TITLE = "Spotify Wrapped"
APP_ICON = "🎧"
PAGE_LAYOUT = "wide"
API_TIMEOUT = 10  # seconds

# Spotify Theme Colors
COLORS = {
    'primary': '#1DB954',
    'secondary': '#1ed760',
    'background': '#191414',
    'card_bg': '#282828',
    'text': '#FFFFFF',
    'text_secondary': '#b3b3b3',
    'error': '#E22134',
    'warning': '#FFA500',
    'success': '#1DB954'
}

# Custom CSS for Spotify theme
SPOTIFY_CSS = f"""
<style>
    /* Main App Background */
    .stApp {{
        background: linear-gradient(135deg, {COLORS['background']} 0%, #1a1a1a 100%);
    }}
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: {COLORS['primary']} !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }}
    
    /* Text */
    p, li, label, span {{
        color: {COLORS['text']} !important;
    }}
    
    /* Buttons */
    .stButton>button {{
        background-color: {COLORS['primary']};
        color: white;
        border-radius: 30px;
        padding: 10px 30px;
        font-weight: bold;
        border: none;
        font-size: 16px;
        transition: all 0.3s ease;
    }}
    
    .stButton>button:hover {{
        background-color: {COLORS['secondary']};
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(29, 185, 84, 0.4);
    }}
    
    /* File Uploader */
    .stFileUploader {{
        background-color: {COLORS['card_bg']};
        border-radius: 10px;
        padding: 20px;
        border: 2px dashed {COLORS['primary']};
    }}
    
    /* Cards */
    .stAlert {{
        background-color: {COLORS['card_bg']};
        border-radius: 10px;
        border-left: 4px solid {COLORS['primary']};
    }}
    
    /* Metrics */
    [data-testid="stMetricValue"] {{
        color: {COLORS['primary']} !important;
        font-size: 32px !important;
    }}
    
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: #000000;
    }}
    
    section[data-testid="stSidebar"] > div {{
        background-color: #000000;
    }}
    
    /* Progress Bar */
    .stProgress > div > div > div > div {{
        background-color: {COLORS['primary']};
    }}
    
    /* Selectbox */
    .stSelectbox {{
        background-color: {COLORS['card_bg']};
    }}
    
    /* Text Input */
    .stTextInput > div > div > input {{
        background-color: {COLORS['card_bg']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['primary']};
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        background-color: {COLORS['card_bg']};
        border-radius: 10px;
    }}
    
    /* Custom Track Card */
    .track-card {{
        background-color: {COLORS['card_bg']};
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid {COLORS['primary']};
        transition: all 0.3s ease;
    }}
    
    .track-card:hover {{
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(29, 185, 84, 0.3);
    }}
    
    /* Feature Card */
    .feature-card {{
        background-color: {COLORS['card_bg']};
        border-radius: 10px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {COLORS['background']};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {COLORS['primary']};
        border-radius: 5px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['secondary']};
    }}
</style>
"""

# Feature emojis
FEATURE_EMOJIS = {
    'top_songs': '🎵',
    'top_artists': '🎤',
    'playlist_age': '📅',
    'listening_age': '🕰️',
    'temporal': '📈',
    'audio_features': '🎨',
    'mood_analysis': '😊',
    'popularity': '⭐',
    'mood_radar': '🎯'
}

# Mood colors
MOOD_COLORS = {
    'Happy': '#FFD700',
    'Sad': '#4169E1',
    'Energetic': '#FF6347',
    'Chill': '#7FFFD4'
}

# Mood emojis
MOOD_EMOJIS = {
    'Happy': '😊',
    'Sad': '😢',
    'Energetic': '⚡',
    'Chill': '😌'
}

# Page titles
PAGE_TITLES = {
    'home': 'Spotify Wrapped',
    'wrapped': 'Your Wrapped',
    'recommendations': 'Music Recommendations'
}

# Error messages
ERROR_MESSAGES = {
    'no_data': '⚠️ Please upload your playlist data first!',
    'api_offline': '❌ Cannot connect to API. Please ensure Flask backend is running on port 5000.',
    'invalid_file': '❌ Invalid file type. Please upload a CSV file.',
    'upload_failed': '❌ Upload failed. Please try again.',
    'no_tracks': '⚠️ No tracks found in the uploaded file.'
}

# Success messages
SUCCESS_MESSAGES = {
    'upload': '✅ Upload successful!',
    'data_loaded': '✅ Your playlist data is loaded!',
    'api_connected': '✅ API Connected'
}

# Add after the existing SPOTIFY_CSS variable (around line 100+)

# Wrapped Card Configuration
WRAPPED_CARD_CONFIG = {
    'aspect_ratio': '9:16',  # Portrait mode like Instagram stories
    'width': '450px',  # Adjust as needed
    'height': '800px',  # 16/9 * 450 = 800
    'border_radius': '20px',
    'font_family': 'Dela Gothic One, sans-serif',
    'font_size': '20px',
    'text_align': 'center'
}

# CSS for Wrapped Cards with Custom Background
WRAPPED_CARD_CSS = """
<style>
    /* Import Dela Gothic One font */
    @import url('https://fonts.googleapis.com/css2?family=Dela+Gothic+One&display=swap');
    
    /* Wrapped Card Container */
    .wrapped-card {
        width: 450px;
        height: 800px;
        margin: 20px auto;
        border-radius: 20px;
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        padding: 40px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        position: relative;
    }
    
    /* Text inside Wrapped Card */
    .wrapped-card h1,
    .wrapped-card h2,
    .wrapped-card h3,
    .wrapped-card p,
    .wrapped-card span {
        font-family: 'Dela Gothic One', sans-serif !important;
        font-size: 20px !important;
        text-align: center !important;
        color: white !important;
        text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.8);
        margin: 10px 0;
    }
    
    /* Title text - larger */
    .wrapped-card .card-title {
        font-size: 32px !important;
        margin-bottom: 20px;
    }
    
    /* Overlay for better text readability */
    .wrapped-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(rgba(0,0,0,0.3), rgba(0,0,0,0.3));
        border-radius: 20px;
        z-index: 1;
    }
    
    /* Ensure content is above overlay */
    .wrapped-card > * {
        position: relative;
        z-index: 2;
    }
</style>
"""