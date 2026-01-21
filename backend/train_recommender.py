"""
Train and save the Spotify music recommender model
Run this ONCE before starting the Flask API
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pickle
import os
import sys

# Add parent directory to path to import from ml/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*70)
print("SPOTIFY RECOMMENDER MODEL TRAINING")
print("="*70)

# Configuration
DATA_PATH = r"D:\codes\Datasets\spotify_data.csv"  # Output from offline_pipeline.py
ML_DIR = r"D:\codes\Spotify_Wrapped\ml" # Save models here

# Create ml directory if it doesn't exist
os.makedirs(ML_DIR, exist_ok=True)

print("\n[1/5] Loading dataset...")
try:
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {len(df)} tracks")
    print(f"   Columns: {list(df.columns)}")
except FileNotFoundError:
    print(f"❌ ERROR: '{DATA_PATH}' not found!")
    print("   Please run 'offline_pipeline.py' first to generate this file.")
    sys.exit(1)

# Audio features for similarity matching
feature_cols = ['valence', 'acousticness', 'danceability', 'energy', 
                'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo']

print("\n[2/5] Preparing features...")

# Check if all required columns exist (case-insensitive)
df.columns = df.columns.str.lower()  # Normalize to lowercase
missing_cols = [col for col in feature_cols if col not in df.columns]

if missing_cols:
    print(f"❌ ERROR: Missing columns: {missing_cols}")
    sys.exit(1)

# Extract features
features = df[feature_cols].copy()

# Handle missing values
features = features.fillna(features.mean())
print(f"✅ Extracted {len(feature_cols)} audio features")

print("\n[3/5] Scaling features...")
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
print("✅ Features normalized using StandardScaler")

print("\n[4/5] Training KNN model...")
knn = NearestNeighbors(n_neighbors=50, metric='euclidean', algorithm='auto')
knn.fit(scaled_features)
print("✅ KNN model trained (k=50, euclidean distance)")

print("\n[5/5] Saving models to ml/ directory...")

# Save KNN model
knn_path = os.path.join(ML_DIR, "recommender_knn.pkl")
with open(knn_path, 'wb') as f:
    pickle.dump(knn, f)
print(f"✅ Saved KNN model: {knn_path}")

# Save scaler
scaler_path = os.path.join(ML_DIR, "recommender_scaler.pkl")
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✅ Saved scaler: {scaler_path}")

# Save processed data (for track lookup)
data_dict = {
    'df': df,
    'feature_cols': feature_cols,
    'scaled_features': scaled_features
}
data_path = os.path.join(ML_DIR, "recommender_data.pkl")
with open(data_path, 'wb') as f:
    pickle.dump(data_dict, f)
print(f"✅ Saved dataset: {data_path}")

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print(f"\nModels saved in: {os.path.abspath(ML_DIR)}")
print("\nNext steps:")
print("1. Start Flask API: python backend/spotify_api_dynamic.py")
print("2. Start Streamlit: streamlit run frontend/streamlit_app.py")
print("="*70)
