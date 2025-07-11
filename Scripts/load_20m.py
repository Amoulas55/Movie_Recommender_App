import pandas as pd

# Correct paths to the extracted files
ratings_path = "/home/u762545/movie_recommender/ratings.csv"
movies_path = "/home/u762545/movie_recommender/movies.csv"

# Load ratings
print("🔄 Loading ratings...")
ratings = pd.read_csv(ratings_path)
print(f"✅ Ratings shape: {ratings.shape}")

# Load movies
print("🔄 Loading movies...")
movies = pd.read_csv(movies_path)
print(f"✅ Movies shape: {movies.shape}")

# Merge on movieId
df = pd.merge(ratings, movies, on="movieId")
print("✅ Merged dataset shape:", df.shape)
print("🧾 Columns:", df.columns.tolist())
print(df.head())

# Save to CSV
df.to_csv("merged_movielens_20m.csv", index=False)
print("💾 Saved as merged_movielens_20m.csv")
