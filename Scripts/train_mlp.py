import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

# 📁 Load data
train_df = pd.read_csv("train_debug.csv")
val_df = pd.read_csv("val_debug.csv")

with open("user2idx_debug.json") as f:
    user2idx = json.load(f)
with open("movie2idx_debug.json") as f:
    movie2idx = json.load(f)

num_users = len(user2idx)
num_movies = len(movie2idx)

# 🎯 Hyperparameters
EMBEDDING_DIM = 64
HIDDEN_DIMS = [128, 64]
BATCH_SIZE = 1024
EPOCHS = 10
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 🧺 PyTorch Dataset
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

train_loader = DataLoader(RatingsDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(RatingsDataset(val_df), batch_size=BATCH_SIZE)

# 🧠 MLP Model
class MLPRecommender(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim, hidden_dims):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embedding_dim)
        self.movie_embed = nn.Embedding(num_movies, embedding_dim)

        layers = []
        input_dim = embedding_dim * 2
        for hdim in hidden_dims:
            layers.append(nn.Linear(input_dim, hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hdim
        layers.append(nn.Linear(input_dim, 1))  # Output: predicted rating

        self.mlp = nn.Sequential(*layers)

    def forward(self, user_ids, movie_ids):
        u = self.user_embed(user_ids)
        m = self.movie_embed(movie_ids)
        x = torch.cat([u, m], dim=1)
        return self.mlp(x).squeeze()

model = MLPRecommender(num_users, num_movies, EMBEDDING_DIM, HIDDEN_DIMS).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 🏋️ Training Loop
def evaluate(loader):
    model.eval()
    total_loss, total_samples = 0, 0
    with torch.no_grad():
        for users, movies, ratings in loader:
            users, movies, ratings = users.to(DEVICE), movies.to(DEVICE), ratings.to(DEVICE)
            preds = model(users, movies)
            loss = criterion(preds, ratings)
            total_loss += loss.item() * len(ratings)
            total_samples += len(ratings)
    rmse = np.sqrt(total_loss / total_samples)
    return rmse

print(f"🚀 Training on {DEVICE}")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for users, movies, ratings in tqdm(train_loader, desc=f"Epoch {epoch}"):
        users, movies, ratings = users.to(DEVICE), movies.to(DEVICE), ratings.to(DEVICE)
        optimizer.zero_grad()
        preds = model(users, movies)
        loss = criterion(preds, ratings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(ratings)

    train_rmse = np.sqrt(total_loss / len(train_loader.dataset))
    val_rmse = evaluate(val_loader)
    print(f"📉 Epoch {epoch}: Train RMSE = {train_rmse:.4f} | Val RMSE = {val_rmse:.4f}")

# 💾 Save model
torch.save(model.state_dict(), "mlp_recommender.pt")
print("✅ Model saved to 'mlp_recommender.pt'")
