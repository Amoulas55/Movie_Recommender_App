import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
from tqdm import tqdm

sns.set(style="whitegrid")
eda_dir = "eda_20m"
os.makedirs(eda_dir, exist_ok=True)

# Path to full dataset
data_path = "merged_movielens_20m.csv"
chunk_size = 1_000_000

# Trackers
rating_counter = Counter()
genre_counter = Counter()
movie_counter = Counter()
user_counter = Counter()

print("🔄 Starting chunked EDA...")
for chunk in tqdm(pd.read_csv(data_path, chunksize=chunk_size)):
    # Rating distribution
    rating_counter.update(chunk["rating"].round(1).value_counts().to_dict())

    # Genre counts
    chunk['genres'] = chunk['genres'].fillna('').astype(str)
    exploded = chunk.assign(genres=chunk['genres'].str.split('|')).explode('genres')
    genre_counter.update(exploded['genres'].value_counts().to_dict())

    # Most rated movies
    movie_counter.update(chunk['title'].value_counts().to_dict())

    # User activity
    user_counter.update(chunk['userId'].value_counts().to_dict())

print("✅ Finished reading chunks.")

# ---------------- PLOTS ----------------
def save_plot(data, title, xlabel, filename, kind="barh", top_n=20):
    plt.figure(figsize=(10, 6))
    if kind == "barh":
        sns.barplot(x=list(data.values())[:top_n], y=list(data.keys())[:top_n], palette="viridis")
    else:
        sns.barplot(x=list(data.keys()), y=list(data.values()), palette="viridis")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(f"{eda_dir}/{filename}")
    plt.close()

# Rating Distribution
rating_dist = dict(sorted(rating_counter.items()))
save_plot(rating_dist, "Rating Distribution", "Count", "rating_distribution.png", kind="bar")

# Genre Popularity
genre_dist = dict(sorted(genre_counter.items(), key=lambda x: x[1], reverse=True))
save_plot(genre_dist, "Genre Popularity", "Number of Ratings", "genre_popularity.png")

# Most Rated Movies
top_movies = dict(sorted(movie_counter.items(), key=lambda x: x[1], reverse=True))
save_plot(top_movies, "Most Rated Movies", "Number of Ratings", "most_rated_movies.png")

# User Rating Frequency
plt.figure(figsize=(10,5))
sns.histplot(list(user_counter.values()), bins=50, color='green')
plt.title("User Rating Activity")
plt.xlabel("Ratings per User")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.savefig(f"{eda_dir}/user_activity_distribution.png")
plt.close()

print("📊 All chunked EDA plots saved in 'eda_20m/' folder.")
