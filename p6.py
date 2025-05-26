"""
pytorch_recommender.py: Two-tower recommender with side features, ranking metrics, margin-ranking loss,
and an actual-vs-predicted comparison table.
Usage:
  python pytorch_recommender.py --epochs 10 --emb-dim 64 --margin 1.0
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
import argparse

class InteractionDataset(Dataset):
    """Yield (u, pos_i, neg_i, side_feats) triples for ranking."""
    def __init__(self, user_ids, item_ids, user_side, num_items):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.user_side = user_side
        self.num_items = num_items

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        u = self.user_ids[idx]
        pos = self.item_ids[idx]
        neg = np.random.randint(self.num_items)
        while neg == pos:
            neg = np.random.randint(self.num_items)
        side = self.user_side[idx]
        return (
            torch.LongTensor([u]),
            torch.LongTensor([pos]),
            torch.LongTensor([neg]),
            torch.FloatTensor(side)
        )

class TwoTowerSide(nn.Module):
    """Two-tower model with side features."""
    def __init__(self, n_users, n_items, side_dim, emb_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.side_mlp = nn.Sequential(
            nn.Linear(side_dim, emb_dim),
            nn.ReLU()
        )

    def forward(self, u, pos, neg, side):
        u_id = self.user_emb(u).squeeze(1)
        pos_e = self.item_emb(pos).squeeze(1)
        neg_e = self.item_emb(neg).squeeze(1)
        u_side = self.side_mlp(side)
        u_e = u_id + u_side
        pos_score = (u_e * pos_e).sum(1)
        neg_score = (u_e * neg_e).sum(1)
        return pos_score, neg_score

def margin_loss(pos_scores, neg_scores, margin=1.0):
    """Margin-ranking (hinge) loss."""
    target = torch.ones_like(pos_scores)
    criterion = nn.MarginRankingLoss(margin=margin)
    return criterion(pos_scores, neg_scores, target)

def evaluate_ranking(model, test_pairs, user_side_full, n_items, k=5, device='cpu'):
    ndcgs, precs = [], []
    model.eval()
    for u in test_pairs['user'].unique():
        true_items = test_pairs[test_pairs['user']==u]['item'].values
        side = torch.FloatTensor(user_side_full[u:u+1]).to(device)
        with torch.no_grad():
            u_id = torch.LongTensor([u]).to(device)
            u_e = model.user_emb(u_id).squeeze(0) + model.side_mlp(side).squeeze(0)
            scores = (u_e @ model.item_emb.weight.t()).cpu().numpy()
        y_true = np.zeros(n_items)
        y_true[true_items] = 1
        order = np.argsort(-scores)
        ndcgs.append(ndcg_score([y_true],[scores], k=k))
        precs.append((y_true[order[:k]]>0).sum()/k)
    print(f"Mean NDCG@{k}: {np.mean(ndcgs):.4f}")
    print(f"Mean Precision@{k}: {np.mean(precs):.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',  type=int,   default=5)
    parser.add_argument('--emb-dim', type=int,   default=32)
    parser.add_argument('--margin',  type=float, default=1.0)
    args = parser.parse_args()

    # Load & preprocess
    df = pd.read_csv('data.csv')
    df = df[df['Frequency']>0]
    u_enc = LabelEncoder(); df['user'] = u_enc.fit_transform(df['Customer_ID'])
    i_enc = LabelEncoder(); df['item'] = i_enc.fit_transform(df['Product'])
    n_users, n_items = df['user'].nunique(), df['item'].nunique()
    df['gender_lbl'] = LabelEncoder().fit_transform(df['Gender'])
    df['age_lbl']    = LabelEncoder().fit_transform(df['Age_Group'])
    df['reg_lbl']    = LabelEncoder().fit_transform(df['Region'])
    scaler = StandardScaler(); df['inc_scaled'] = scaler.fit_transform(df[['Annual_Income']])
    side_cols = ['gender_lbl','age_lbl','reg_lbl','inc_scaled']
    user_side = df[side_cols].values

    # Train/test split
    pairs = df[['user','item']]
    train_df, test_df = train_test_split(pairs, test_size=0.2, random_state=42)
    train_side = user_side[train_df.index]

    # DataLoader
    ds = InteractionDataset(
        train_df['user'].values,
        train_df['item'].values,
        train_side,
        n_items
    )
    loader = DataLoader(ds, batch_size=256, shuffle=True)

    # Model & optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TwoTowerSide(n_users, n_items, side_dim=len(side_cols), emb_dim=args.emb_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training
    for ep in range(args.epochs):
        model.train(); total_loss = 0
        for u,pos,neg,side in loader:
            u,pos,neg,side = u.to(device), pos.to(device), neg.to(device), side.to(device)
            ps, ns = model(u,pos,neg,side)
            loss = margin_loss(ps, ns, margin=args.margin)
            optim.zero_grad(); loss.backward(); optim.step()
            total_loss += loss.item()
        print(f"Epoch {ep+1}/{args.epochs} - Avg Loss: {total_loss/len(loader):.4f}")

    # Evaluation
    evaluate_ranking(model, test_df, user_side, n_items, k=5, device=device)

    # Sample recommendation
    u0 = test_df['user'].iloc[0]
    side0 = torch.FloatTensor(user_side[u0:u0+1]).to(device)
    with torch.no_grad():
        u_id = torch.LongTensor([u0]).to(device)
        u_e = model.user_emb(u_id).squeeze(0) + model.side_mlp(side0).squeeze(0)
        scores = (u_e @ model.item_emb.weight.t()).cpu()
    top5 = torch.topk(scores,5).indices.numpy()
    print("Top-5 for user", u_enc.inverse_transform([u0])[0], ":", i_enc.inverse_transform(top5))

    # -------------------------------
    # Actual vs. Predicted Comparison
    # -------------------------------
    results = []
    for u in test_df['user'].unique():
        # Actual items in test set for user u
        actual_idxs = test_df[test_df['user'] == u]['item'].tolist()
        actual = i_enc.inverse_transform(actual_idxs)

        # Predicted top-5 items
        side = torch.FloatTensor(user_side[u:u+1]).to(device)
        u_id = torch.LongTensor([u]).to(device)
        with torch.no_grad():
            u_e = model.user_emb(u_id).squeeze(0) + model.side_mlp(side).squeeze(0)
            scores = (u_e @ model.item_emb.weight.t()).cpu().numpy()
        top5_idxs = np.argsort(-scores)[:5]
        predicted = i_enc.inverse_transform(top5_idxs)

        results.append({
            'user_id': u_enc.inverse_transform([u])[0],
            'actual_purchased': list(actual),
            'predicted_top5': list(predicted)
        })

    # Build and display DataFrame
    cmp_df = pd.DataFrame(results)
    print("\nActual vs. Predicted Top-5 per User:\n")
    print(cmp_df.to_string(index=False))
