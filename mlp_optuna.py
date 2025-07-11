import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import optuna
import os

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

# 🎯 Objective function
def objective(trial):
    embed_dim = trial.suggest_categorical("embed_dim", [32, 64, 128])
    hidden_dims = []
    num_layers = trial.suggest_int("num_layers", 1, 3)
    for i in range(num_layers):
        hidden_dims.append(trial.suggest_int(f"hidden_dim_{i}", 32, 256))
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    model = MLPRecommender(num_users, num_movies, embed_dim, hidden_dims, dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024)

    best_val_rmse = float("inf")
    patience, wait = 2, 0

    for epoch in range(10):
        model.train()
        for users, movies, ratings in train_loader:
            users, movies, ratings = users.to(DEVICE), movies.to(DEVICE), ratings.to(DEVICE)
            optimizer.zero_grad()
            preds = model(users, movies)
            loss = criterion(preds, ratings)
            loss.backward()
            optimizer.step()

        val_rmse = evaluate(model, val_loader, criterion)
        trial.report(val_rmse, epoch)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    return best_val_rmse

# 🧪 Run Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("✅ Best trial:")
print(study.best_trial)

# 🏁 Final training with best params
best_params = study.best_trial.params
embed_dim = best_params["embed_dim"]
dropout = best_params["dropout"]
lr = best_params["lr"]
hidden_dims = [best_params[f"hidden_dim_{i}"] for i in range(best_params["num_layers"])]

model = MLPRecommender(num_users, num_movies, embed_dim, hidden_dims, dropout).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024)

best_rmse = float("inf")
wait = 0
for epoch in range(30):  # longer training now
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
