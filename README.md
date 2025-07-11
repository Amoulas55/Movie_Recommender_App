# 🎬 Movie Recommender System (MLP + Streamlit)

This project is a movie recommender system built using a Multi-Layer Perceptron (MLP) trained on the MovieLens 20M dataset. The app is deployed using **Streamlit**, providing personalized movie suggestions based on user input, genre filters, and decade preferences.

---

## 📊 Example EDA Visuals

EDA plots such as `genre_popularity.png`, `most_rated_movies.png`, `user_activity_distribution.png`, and `rating_distribution.png` were used for early insights and then removed from the repo for size. You can regenerate them using `advanced_eda.py`.

---

## 🧠 Model Overview

- Model Type: **Multi-Layer Perceptron (MLP)**
- Inputs:
  - `user_id`
  - `movie_id`
- Architecture:
  - Embedding layers for users and movies
  - Fully connected layers: `[512 ➝ 128 ➝ 64 ➝ 1]`
  - Activation: ReLU
  - Dropout: 0.2
- Loss: MSE (Mean Squared Error)
- Optimizer: Adam
- Training: `train_mlp.py` with data processed by `preprocess_movielens.py`
- Final Model: `best_mlp_model.pt` (downloaded via `gdown`)

---

## 🌐 Google Drive Integration

To avoid uploading large CSV/JSON/model files directly, this project uses `gdown` to download all necessary resources from a shared Google Drive folder. This keeps your GitHub repo clean and under size limits.

---

## 🚀 How to Run the App

### 1️⃣ Prerequisites

- Python 3.8+
- Install required libraries:


pip install -r requirements.txt

If you don’t have a requirements.txt, use:
pip install torch pandas numpy streamlit gdown


### 2️⃣ Run the Streamlit App
streamlit run streamlit_app_movie_gdown.py
This will:

✅ Automatically download the model (best_mlp_model.pt) and datasets from Google Drive

🌐 Launch a browser tab with the Streamlit interface

### 3️⃣ Select Your Preferences
🎥 Choose up to 20 movies you’ve enjoyed

🎯 Optionally filter recommendations by genre and decade

✅ View smart recommendations tailored to your preferences

## 🎉 Features
🔍 Uses MLP Embeddings + Cosine Similarity to recommend similar movies

🧠 Streamlit UI with:

🎞️ Movie selection

🧩 Genre and decade filters

🔢 Adjustable number of recommendations

⭐ Popularity-aware scoring to improve result quality

💻 Ready for local use or deployment

## 📂 Google Drive Files Used
These large files are hosted remotely and downloaded automatically at runtime:

best_mlp_model.pt

movie2idx_debug.json

user2idx_debug.json

movies.csv

ratings.csv

You do not need to include them in the repo.

## 📜 License
This project is licensed under the MIT License — see the LICENSE file for details.

## 🤝 Acknowledgements

- **MovieLens 20M Dataset**: This work uses data from the [MovieLens](https://grouplens.org/datasets/movielens/) project. If you use this dataset, please cite:

> F. Maxwell Harper and Joseph A. Konstan. 2015.  
> *The MovieLens Datasets: History and Context*.  
> ACM Transactions on Interactive Intelligent Systems (TiiS), 5(4), Article 19 (December 2015).  
> DOI: [10.1145/2827872](https://doi.org/10.1145/2827872)

- [Streamlit](https://streamlit.io/) for the web-based interface  
- [PyTorch](https://pytorch.org/) for model training  
