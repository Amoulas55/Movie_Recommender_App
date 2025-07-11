import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import optuna

# 📁 Load data
train_df = pd.read_csv("train_debug.csv")
val_df = pd.read_csv("val_debug.csv")

with open("user2idx_debug.json") as f:
    user2idx = json.load(f)
with open("movie2idx_debug.json") as f:
    movie2idx = json.load(f)

num_users = len(user2idx)
num_movies = len(movie2idx)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 🧺 Dataset
class RatingsDataset(Dataset):
    def __init__(self, df):
        self.users = df['user_idx'].values
        self.movies = df['movie_idx'].values
        self.ratings = df['rating'].values.astype(np.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.movies[idx], dtype=torch.long),
            torch.tensor(self.ratings[idx], dtype=torch.float),
        )

train_dataset = RatingsDataset(train_df)
val_dataset = RatingsDataset(val_df)

# 🧠 MLP Recommender
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

# 🔁 Evaluation
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_samples = 0, 0
    with torch.no_grad():
        for users, movies, ratings in loader:
            users, movies, ratings = users.to(DEVICE), movies.to(DEVICE), ratings.to(DEVICE)
            preds = model(users, movies)
            loss = criterion(preds, ratings)
            total_loss += loss.item() * len(ratings)
            total_samples += len(ratings)
    return np.sqrt(total_loss / total_samples)

# 🧪 Use best Optuna params
embed_dim = 256
hidden_dims = [128, 64]
dropout = 0.2
lr = 9.238e-5

model = MLPRecommender(num_users, num_movies, embed_dim, hidden_dims, dropout).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
criterion = nn.MSELoss()
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024)

best_rmse = float("inf")
wait = 0
for epoch in range(50):
    model.train()
    for users, movies, ratings in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        users, movies, ratings = users.to(DEVICE), movies.to(DEVICE), ratings.to(DEVICE)
        optimizer.zero_grad()
        preds = model(users, movies)
        loss = criterion(preds, ratings)
        loss.backward()
        optimizer.step()

    val_rmse = evaluate(model, val_loader, criterion)
    print(f"📉 Epoch {epoch+1}: Val RMSE = {val_rmse:.4f}")

    if val_rmse < best_rmse:
        best_rmse = val_rmse
        wait = 0
        torch.save(model.state_dict(), "best_mlp_model.pt")
        print("💾 Model saved!")
    else:
        wait += 1
        if wait >= 3:
            print("⏹️ Early stopping")
            break

print("✅ Final model saved as 'best_mlp_model.pt'")
