import os
import pickle

from ml.preprocessing import load_data, clean_data, select_features


# ---------------------------------
# Resolve project paths (deployment-safe)
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "backend", "model.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "backend", "scaler.pkl")


# ---------------------------------
# Load model and scaler (once)
# ---------------------------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)


# ---------------------------------
# Shared ML inference function
# ---------------------------------
def predict_mood_distribution():
    """
    Loads data, runs mood prediction,
    and returns top mood + distribution.
    """
    df = load_data()
    df = clean_data(df)

    X = select_features(df)
    X_scaled = scaler.transform(X)

    df["predicted_mood"] = model.predict(X_scaled)

    mood_distribution = df["predicted_mood"].value_counts().to_dict()
    top_mood = max(mood_distribution, key=mood_distribution.get)

    return top_mood, mood_distribution
