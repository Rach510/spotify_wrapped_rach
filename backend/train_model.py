import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from ml.preprocessing import (load_data,clean_data,select_features,create_mood_labels)


# -------------------- DATA PREPARATION --------------------

def prepare_data():
    df = load_data()
    df = clean_data(df)
    df = create_mood_labels(df)

    X = select_features(df)
    y = df["mood"]

    return X, y


# -------------------- TRAIN / TEST SPLIT --------------------

def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=42,
        stratify=y
    )


# -------------------- FEATURE SCALING --------------------

def feature_scaling(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


# -------------------- MODEL TRAINING --------------------

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter = 1000)
    model.fit(X_train, y_train)
    return model


# -------------------- MODEL EVALUATION --------------------

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))


# -------------------- SAVE MODEL & SCALER --------------------

def save_model(model, scaler):
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


# -------------------- MAIN PIPELINE --------------------

if __name__ == "__main__":
    X, y = prepare_data()

    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = feature_scaling(X_train, X_test)

    model = train_model(X_train_scaled, y_train)
    evaluate_model(model, X_test_scaled, y_test)
    save_model(model, scaler)
