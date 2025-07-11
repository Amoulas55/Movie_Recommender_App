import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import gdown
import os

# 📥 Download required files from Google Drive if not present
def download_if_missing(url, output_path):
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

# 🔗 File URLs and paths
files_to_download = {
    "best_mlp_model.pt": "https://drive.google.com/uc?id=1U1Qe4QcjuGkqXwwFuxNpVWSZo5_Bl4qN",
    "movie2idx_debug.json": "https://drive.google.com/uc?id=1b1Fd__l3ITgaQexSuCTjYnhF55ZIdyBQ",
    "movies.csv": "https://drive.google.com/uc?id=1NFXbJIwlR1IDeueC8pTAlhsjzSEErjkO",
    "ratings.csv": "https://drive.google.com/uc?id=1gYZhik2GLnSS0iMvpBfMQVIBNZ2ZzP5v"
}

for filename, url in files_to_download.items():
    download_if_missing(url, filename)

# 📦 Load mappings
with open("movie2idx_debug.json") as f:
    movie2idx = json.load(f)
idx2movie = {v: k for k, v in movie2idx.items()}

# 🏷️ Load movie metadata
movies_df = pd.read_csv("movies.csv")
movie_id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))
movie_id_to_genre = dict(zip(movies_df['movieId'], movies_df['genres']))

# 🔤 Normalize title keys for better matching
def normalize(title):
    return title.lower().replace(", the", "").strip()
normalized_title_map = {normalize(v): k for k, v in movie_id_to_title.items()}

# ⭐ Load ratings for popularity boost
ratings_df = pd.read_csv("ratings.csv")
movie_popularity = ratings_df['movieId'].value_counts().to_dict()
max_pop = max(movie_popularity.values())
popularity_scores = {k: np.log1p(v) / np.log1p(max_pop) for k, v in movie_popularity.items()}

# 🧠 Define model
class MLPRecommender(nn.Module):
    def __init__(self, num_users, num_movies, embed_dim, hidden_dims, dropout):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.movie_embed = nn.Embedding(num_movies, embed_dim)
        layers = []
        input_dim = embed_dim * 2
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_ids, movie_ids):
        u = self.user_embed(user_ids)
        m = self.movie_embed(movie_ids)
        x = torch.cat([u, m], dim=1)
        return self.mlp(x).squeeze()

# ⚙️ Hyperparameters (from best model)
EMBED_DIM = 256
HIDDEN_DIMS = [128, 64]
DROPOUT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MLPRecommender(num_users=138493, num_movies=len(movie2idx), embed_dim=EMBED_DIM, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT).to(DEVICE)
model.load_state_dict(torch.load("best_mlp_model.pt", map_location=DEVICE))
model.eval()

# 🌟 Streamlit UI
st.title("🍿 Movie Recommender")
st.markdown("Pick a few movies you like, and we'll suggest similar ones!")

num_recs = st.slider("How many recommendations?", min_value=5, max_value=20, value=5)
all_genres = sorted(set('|'.join(movies_df['genres']).split('|')))
genre_filters = st.multiselect("Filter by genre (optional):", all_genres)
decade_filter = st.selectbox("Filter by decade (from selected onwards):", ["All"] + [f"{y}s" for y in range(1900, 2030, 10)])

available_titles = sorted([movie_id_to_title.get(int(mid), f"Movie {mid}") for mid in movie2idx.keys()])
selected_titles = st.multiselect("🤖 Pick movies you liked:", options=available_titles, max_selections=10)

if selected_titles:
    temp_user_id = torch.tensor([138492]).to(DEVICE)
    liked_movie_ids = [k for k, v in movie_id_to_title.items() if v in selected_titles]
    liked_movie_idxs = [movie2idx[str(mid)] for mid in liked_movie_ids if str(mid) in movie2idx]

    with torch.no_grad():
        movie_embeddings = model.movie_embed.weight.detach()
        liked_movie_idxs_tensor = torch.tensor(liked_movie_idxs).to(DEVICE)
        liked_embeddings = movie_embeddings[liked_movie_idxs_tensor]
        user_embedding = liked_embeddings.mean(dim=0, keepdim=True)

        normalized_movies = movie_embeddings / movie_embeddings.norm(dim=1, keepdim=True)
        normalized_user = user_embedding / user_embedding.norm(dim=1, keepdim=True)
        cosine_sim = torch.matmul(normalized_user, normalized_movies.T).squeeze()

        for idx in liked_movie_idxs:
            cosine_sim[idx] = -float('inf')

        scored_candidates = []
        for i in torch.argsort(cosine_sim, descending=True).tolist():
            movie_id = int(idx2movie[i])
            title = movie_id_to_title.get(movie_id, "Unknown")
            genres = movie_id_to_genre.get(movie_id, "")
            genre_list = genres.split('|')
            year = int(title[-5:-1]) if title[-1] == ")" and title[-6] == "(" else None

            if genre_filters:
                if not set(genre_filters).intersection(set(genre_list)):
                    continue

            if decade_filter != "All" and year:
                decade_start = int(decade_filter[:4])
                if year < decade_start:
                    continue

            if movie_popularity.get(movie_id, 0) < 100:
                continue

            popularity = popularity_scores.get(movie_id, 0.0)
            final_score = cosine_sim[i].item() * (0.5 + 0.5 * popularity)

            scored_candidates.append((final_score, title, genres))
            if len(scored_candidates) >= num_recs * 2:
                break

        final_recommendations = sorted(scored_candidates, key=lambda x: x[0], reverse=True)[:num_recs]

        st.subheader("📽️ Top Recommendations for You:")
        if final_recommendations:
            for _, title, genres in final_recommendations:
                st.write(f"✅ {title} [{genres}]")
        else:
            st.write("⚠️ No recommendations found. Try removing filters or choosing more movies.")
