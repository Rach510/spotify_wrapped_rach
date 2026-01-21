import pandas as pd
import numpy as np

import os
import pandas as pd
import numpy as np

# -----------------------------
# Path-safe data loader
# -----------------------------
def load_data():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "..", "data", "data.csv")
    df = pd.read_csv(data_path)
    return df

if __name__ =="__main__":
     df=load_data()
     print(df.head())
     print("shape:",df.shape)

     print("columns:",df.columns)



FEATURES = [
    'danceability',
    'energy',
    'loudness',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo'
]

def select_features (df):
     return df[FEATURES]

def clean_data(df):
     df=df.dropna()
     return df     # the ML project cannot handle empty values
                   # and since dataset is large, it wouldnt make much impact



# NOW WE WILL MAKE MOOD PREFERENCE LOGIC 
# MOODS - HAPPY, CHILL, ENERGETIC, SAD
# happy : high valence + high energy
# valence : low energy + high acousticness
# energetic : high energy + high tempo
# sad : low valence + low energy

def assign_mood(row):
     if row['valence']>0.6 and row['energy']>0.6:
          return 'happy'
     elif row['energy'] < 0.4 and row['acousticness'] > 0.5:
        return 'Chill'
     elif row['energy'] > 0.7 and row['tempo'] > 120:
         return 'Energetic'
     else:
        return 'Sad'

def create_mood_labels(df):
     df['mood']=df.apply(assign_mood,axis=1)
     return df


if __name__ =="__main__":
     df = load_data()
     df = clean_data(df)
     df = create_mood_labels(df)
     X = select_features(df)

     y = df['mood']

     print("Features shape:", X.shape)
     print("Mood distribution:")
     print(y.value_counts())