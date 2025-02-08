from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import beta

app = Flask(__name__)

# Load dataset
df = pd.read_csv("high_popularity_spotify_data.csv")

# Bersihkan data
df_cleaned = df.dropna()
df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
df_cleaned = df_cleaned.dropna()

# Preprocessing
numerical_features = ['energy', 'tempo', 'danceability', 'loudness', 'valence', 'speechiness', 'instrumentalness', 'acousticness']
categorical_features = ['playlist_genre', 'playlist_subgenre']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

processed_data = preprocessor.fit_transform(df_cleaned)

# Reduksi dimensi dengan PCA
pca = PCA(n_components=10)
reduced_data = pca.fit_transform(processed_data.toarray())
similarity_matrix = cosine_similarity(reduced_data)

# Thompson Sampling Recommender
class ThompsonSamplingRecommender:
    def __init__(self, similarity_scores, threshold=0.7):
        self.similarity_scores = similarity_scores
        self.threshold = threshold
        self.alpha = np.ones_like(similarity_scores)
        self.beta = np.ones_like(similarity_scores)
        self.total_recommendations_per_song = np.zeros_like(similarity_scores)  # Total rekomendasi per lagu
        self.total_rewards_per_song = np.zeros_like(similarity_scores)          # Total reward per lagu
    
    def recommend(self, input_song_idx, n_recommendations=5, n_iterations=1000):
        for _ in range(n_iterations):
            sampled_probs = np.random.beta(self.alpha[input_song_idx], self.beta[input_song_idx])
            sampled_probs[input_song_idx] = -np.inf
            recommended_song = np.argmax(sampled_probs)
            
            similarity = self.similarity_scores[input_song_idx, recommended_song]
            
            # Update statistik per lagu
            self.total_recommendations_per_song[input_song_idx, recommended_song] += 1
            if similarity > self.threshold:
                self.alpha[input_song_idx, recommended_song] += 1
                self.total_rewards_per_song[input_song_idx, recommended_song] += 1
            else:
                self.beta[input_song_idx, recommended_song] += 1
        
        recommendations = np.argsort(-self.alpha[input_song_idx])[1:n_recommendations+1]
        return recommendations

recommender = ThompsonSamplingRecommender(similarity_matrix)

# Route untuk halaman utama
@app.route("/")
def home():
    return render_template("index.html", songs=df_cleaned[['track_name', 'track_artist']].to_dict('records'))

# Route untuk mencari lagu berdasarkan kata kunci
@app.route("/search", methods=["GET"])
def search_songs():
    query = request.args.get("query", "").lower()
    if not query:
        return jsonify([])
    
    # Filter lagu yang cocok dengan query (baik di judul atau artis)
    matched_songs = df_cleaned[
        (df_cleaned['track_name'].str.lower().str.contains(query)) |
        (df_cleaned['track_artist'].str.lower().str.contains(query))
    ].head(5)  # Ambil 5 hasil teratas
    
    results = matched_songs[['track_name', 'track_artist']].to_dict('records')
    return jsonify(results)

# Route untuk mendapatkan rekomendasi
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    song_name = data.get("song_name")
    song_artist = data.get("song_artist")
    
    song_idx = df_cleaned[(df_cleaned['track_name'] == song_name) & (df_cleaned['track_artist'] == song_artist)].index[0]
    
    recommendations = recommender.recommend(song_idx)
    
    results = []
    for idx in recommendations:
        results.append({
            "track_name": df_cleaned.iloc[idx]['track_name'],
            "track_artist": df_cleaned.iloc[idx]['track_artist'],
            "uri": df_cleaned.iloc[idx]['uri'].split(":")[-1],  
            "similarity": float(similarity_matrix[song_idx, idx]),
            "total_recommended": int(recommender.total_recommendations_per_song[song_idx, idx]),
            "total_reward": int(recommender.total_rewards_per_song[song_idx, idx])
        })
    
    return jsonify({
        "input_song": f"{song_name} oleh {song_artist}",
        "recommendations": results
    })

if __name__ == "__main__":
    app.run(debug=True)