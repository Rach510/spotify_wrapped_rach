import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pickle
import os
import ast
import warnings
warnings.filterwarnings('ignore')

class SpotifyMusicRecommender:
    def __init__(self, data_path=None):
        """Initialize the recommender system with the dataset."""
        if data_path:
            print("Loading dataset...")
            self.df = pd.read_csv(data_path)
            print(f"Loaded {len(self.df)} songs")
            
            # Audio features to use for similarity
            self.feature_cols = ['valence', 'acousticness', 'danceability', 'energy', 
                                'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo']
            
            # Prepare the data
            self.prepare_data()
    
    def prepare_data(self):
        """Prepare and scale the feature data."""
        print("Preparing features...")
        
        # Normalize column names to lowercase
        self.df.columns = self.df.columns.str.lower()
        
        # Extract features
        self.features = self.df[self.feature_cols].copy()
        
        # Handle any missing values
        self.features = self.features.fillna(self.features.mean())
        
        # Scale features
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.features)
        
        # Build KNN model
        print("Building KNN model...")
        self.knn = NearestNeighbors(n_neighbors=50, metric='euclidean')
        self.knn.fit(self.scaled_features)
    
    def save_model(self, model_dir='../ml/'):
        """
        Save the trained model, scaler, and data to files.
        
        Parameters:
        -----------
        model_dir : str
            Directory to save model files (default: '../ml/')
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save KNN model
        knn_path = os.path.join(model_dir, 'recommender_knn.pkl')
        with open(knn_path, 'wb') as f:
            pickle.dump(self.knn, f)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, 'recommender_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save data
        data_dict = {
            'df': self.df,
            'feature_cols': self.feature_cols,
            'scaled_features': self.scaled_features
        }
        data_path = os.path.join(model_dir, 'recommender_data.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump(data_dict, f)
        
        print(f"✅ Model saved to {model_dir}")
    
    @classmethod
    def load_model(cls, model_dir='../ml/'):
        """
        Load a pre-trained model from files.
        
        Parameters:
        -----------
        model_dir : str
            Directory containing model files (default: '../ml/')
        
        Returns:
        --------
        SpotifyMusicRecommender
            Loaded recommender instance ready to use
        """
        print(f"Loading pre-trained model from {model_dir}...")
        
        # Create instance without initializing
        instance = cls.__new__(cls)
        
        # Load KNN model
        knn_path = os.path.join(model_dir, 'recommender_knn.pkl')
        with open(knn_path, 'rb') as f:
            instance.knn = pickle.load(f)
        
        # Load scaler
        scaler_path = os.path.join(model_dir, 'recommender_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            instance.scaler = pickle.load(f)
        
        # Load data
        data_path = os.path.join(model_dir, 'recommender_data.pkl')
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        instance.df = data_dict['df']
        instance.feature_cols = data_dict['feature_cols']
        instance.scaled_features = data_dict['scaled_features']
        
        print(f"✅ Model loaded successfully ({len(instance.df)} tracks)")
        return instance
    
    def get_random_songs(self, n=10):
        """Get n random songs for rating."""
        random_indices = np.random.choice(len(self.df), size=min(n, len(self.df)), replace=False)
        return self.df.iloc[random_indices].copy()
    
    def recommend_from_ratings(self, rated_indices, ratings, n_recommendations=10):
        """
        Recommend songs based on user ratings using weighted KNN.
        
        Parameters:
        -----------
        rated_indices : list
            List of dataframe indices for rated songs
        ratings : list or array
            Ratings (1-5) corresponding to each song in rated_indices
        n_recommendations : int
            Number of recommendations to return
        
        Returns:
        --------
        DataFrame
            Recommended tracks with similarity scores
        """
        # Convert to numpy array
        ratings = np.array(ratings)
        
        # Normalize ratings to use as weights (higher rating = higher weight)
        # We give more weight to higher rated songs
        weights = ratings / ratings.sum()
        
        # Get scaled features for rated songs
        rated_features = self.scaled_features[rated_indices]
        
        # Calculate weighted average of rated songs' features
        # This creates a "user profile" based on their preferences
        weighted_profile = np.average(rated_features, axis=0, weights=weights)
        
        # Find similar songs using KNN
        # We get more neighbors than needed to filter out rated songs
        distances, indices = self.knn.kneighbors(
            [weighted_profile], 
            n_neighbors=min(100, len(self.df))
        )
        
        # Filter out songs that were already rated
        recommendations = []
        recommendation_distances = []
        
        for idx, dist in zip(indices[0], distances[0]):
            if idx not in rated_indices:
                recommendations.append(idx)
                recommendation_distances.append(dist)
            if len(recommendations) >= n_recommendations:
                break
        
        # Get recommended songs
        recommended_songs = self.df.iloc[recommendations].copy()
        
        # Add similarity score (convert distance to similarity: 1 - normalized_distance)
        max_dist = max(recommendation_distances) if recommendation_distances else 1
        recommended_songs['similarity_score'] = [
            1 - (dist / max_dist) for dist in recommendation_distances
        ]
        
        return recommended_songs
    
    def display_song(self, song, index=None):
        """Display song information in a readable format."""
        if index is not None:
            print(f"\n{index}.")
        
        # Try to get track name (handle different column names)
        track_name = song.get('track_name') or song.get('name', 'Unknown')
        
        # Parse artists if it's a string representation of a list
        artists = song.get('artists') or song.get('artist_name(s)', 'Unknown')
        if isinstance(artists, str):
            try:
                artists = ast.literal_eval(artists)
                if isinstance(artists, list):
                    artists = ', '.join(artists)
            except:
                pass
        
        print(f"  Song: {track_name}")
        print(f"  Artist(s): {artists}")
        
        if 'year' in song:
            print(f"  Year: {int(song['year'])}")
        
        if 'popularity' in song:
            print(f"  Popularity: {song['popularity']}")


def main():
    """Main function to run the recommender system."""
    data_path = "spotify_data.csv"
    
    # Initialize the recommender
    recommender = SpotifyMusicRecommender(data_path)
    
    print("\n" + "="*70)
    print("MUSIC RECOMMENDATION SYSTEM - RATING BASED")
    print("="*70)
    print("\nYou will rate 10 random songs on a scale of 1-5:")
    print("  1 = Don't like at all")
    print("  2 = Don't really like")
    print("  3 = It's okay")
    print("  4 = Like it")
    print("  5 = Love it")
    print("="*70)
    
    # Get random songs
    random_songs = recommender.get_random_songs(10)
    ratings = []
    rated_indices = random_songs.index.tolist()
    
    # Get ratings
    for idx, (_, song) in enumerate(random_songs.iterrows(), 1):
        recommender.display_song(song, idx)
        
        while True:
            try:
                rating = input("  Your rating (1-5): ").strip()
                rating = int(rating)
                if 1 <= rating <= 5:
                    ratings.append(rating)
                    break
                else:
                    print("  Please enter a number between 1 and 5.")
            except ValueError:
                print("  Please enter a valid number.")
    
    # Generate recommendations
    print("\n" + "="*70)
    print("GENERATING PERSONALIZED RECOMMENDATIONS...")
    print("="*70)
    
    recommendations = recommender.recommend_from_ratings(
        rated_indices, 
        ratings, 
        n_recommendations=10
    )
    
    # Display recommendations
    print("\n🎵 YOUR PERSONALIZED RECOMMENDATIONS 🎵\n")
    
    for idx, (_, song) in enumerate(recommendations.iterrows(), 1):
        recommender.display_song(song, idx)
        if 'similarity_score' in song:
            print(f"  Match Score: {song['similarity_score']:.2%}")
    
    print("\n" + "="*70)
    print("Enjoy your personalized music recommendations!")
    print("="*70)


if __name__ == "__main__":
    main()
