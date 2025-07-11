import pandas as pd
import numpy as np
import json
from tqdm import tqdm

# Load full dataset
df = pd.read_csv("merged_movielens_20m.csv")
print(f"✅ Original shape: {df.shape}")

# Filter users and movies with <5 ratings
min_ratings = 5
user_counts = df['userId'].value_counts()
df = df[df['userId'].isin(user_counts[user_counts >= min_ratings].index)]

movie_counts = df['movieId'].value_counts()
df = df[df['movieId'].isin(movie_counts[movie_counts >= min_ratings].index)]

print(f"✅ After filtering: {df.shape}")

# Encode userId and movieId
unique_users = df['userId'].unique()
unique_movies = df['movieId'].unique()

user2idx = {int(uid): idx for idx, uid in enumerate(sorted(unique_users))}
movie2idx = {int(mid): idx for idx, mid in enumerate(sorted(unique_movies))}

df['user_idx'] = df['userId'].map(user2idx)
df['movie_idx'] = df['movieId'].map(movie2idx)

# Keep only relevant columns
df_model = df[['user_idx', 'movie_idx', 'rating']]

# Split per user
train_rows, val_rows, test_rows = [], [], []

print("🔄 Splitting train/val/test per user...")
for user_id, user_data in tqdm(df_model.groupby('user_idx')):
    user_data = user_data.sample(frac=1, random_state=42)  # Shuffle
    n = len(user_data)
    if n < 3:
        train_rows.append(user_data)
    else:
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        train_rows.append(user_data.iloc[:train_end])
        val_rows.append(user_data.iloc[train_end:val_end])
        test_rows.append(user_data.iloc[val_end:])

train_df = pd.concat(train_rows)
val_df = pd.concat(val_rows)
test_df = pd.concat(test_rows)

print(f"✅ Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

# Save datasets
train_df.to_csv("train_debug.csv", index=False)
val_df.to_csv("val_debug.csv", index=False)
test_df.to_csv("test_debug.csv", index=False)

# Save mappings (keys must be strings)
user2idx_str = {str(k): v for k, v in user2idx.items()}
movie2idx_str = {str(k): v for k, v in movie2idx.items()}

with open("user2idx_debug.json", "w") as f:
    json.dump(user2idx_str, f)
with open("movie2idx_debug.json", "w") as f:
    json.dump(movie2idx_str, f)

print("💾 Saved: train_debug.csv, val_debug.csv, test_debug.csv, user2idx_debug.json, movie2idx_debug.json")
