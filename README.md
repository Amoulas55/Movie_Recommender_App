
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

```bash
pip install -r requirements.txt
