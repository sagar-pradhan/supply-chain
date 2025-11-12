"""
Supply Chain Graph Neural Network (GCN) Pipeline
-------------------------------------------------
Author: Sagar Pradhan
Description:
    Full implementation of a Graph Convolutional Network (GCN)
    for supply chain production prediction using PyTorch Geometric.

    This script is designed to run directly in Kaggle:
    - Automatically installs PyTorch Geometric dependencies if missing
    - Loads supply chain node/edge/production data
    - Builds graph data using PyG
    - Trains a 2-layer GCN
    - Evaluates and saves model outputs & visualizations
"""

# ==========================================================
# 0. SETUP & INSTALLATION
# ==========================================================
import sys
import subprocess
import importlib
import os
import time

# Function to ensure PyTorch Geometric is installed
def ensure_pyg():
    try:
        import torch_geometric
        return True
    except Exception:
        print("⚙️ Installing PyTorch Geometric (may take ~1–2 minutes)...")
        cmd = [
            sys.executable, "-m", "pip", "install", "-q",
            "torch-scatter", "torch-sparse", "torch-cluster",
            "torch-spline-conv", "torch-geometric",
            "-f", "https://data.pyg.org/whl/torch-2.6.0+cu124.html"
        ]
        subprocess.check_call(cmd)
        import torch_geometric
        return True

# Try installing PyG if needed
try:
    ensure_pyg()
except Exception as e:
    raise RuntimeError(f"PyG installation failed. Restart the kernel and rerun. Error: {e}")

# ==========================================================
# 1. IMPORTS
# ==========================================================
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ==========================================================
# 2. SEED SETUP
# ==========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==========================================================
# 3. FILE PATHS (UPDATE IF NEEDED)
# ==========================================================
NODES_CSV = "/kaggle/input/supplygraph-supply-chain-planning-using-gnns/Raw Dataset/Nodes/Nodes.csv"
EDGES_CSV = "/kaggle/input/supplygraph-supply-chain-planning-using-gnns/Raw Dataset/Edges/Edges (Plant).csv"
PROD_CSV  = "/kaggle/input/supplygraph-supply-chain-planning-using-gnns/Raw Dataset/Temporal Data/Unit/Production .csv"

# ==========================================================
# 4. LOAD CSVs
# ==========================================================
nodes_df = pd.read_csv(NODES_CSV)
edges_df = pd.read_csv(EDGES_CSV)
prod_df  = pd.read_csv(PROD_CSV)
print(f"✅ Loaded data | Nodes: {nodes_df.shape} | Edges: {edges_df.shape} | Production: {prod_df.shape}")

# ==========================================================
# 5. NODE MAPPING & EDGE INDEX
# ==========================================================
if 'Node' not in nodes_df.columns:
    raise RuntimeError("❌ 'Nodes.csv' must have a 'Node' column.")

node_ids = nodes_df['Node'].astype(str).tolist()
id2idx = {nid: i for i, nid in enumerate(node_ids)}
num_nodes = len(node_ids)

if not {'node1', 'node2'}.issubset(edges_df.columns):
    raise RuntimeError("❌ 'Edges.csv' must contain 'node1' and 'node2' columns.")

edges_src = edges_df['node1'].astype(str).map(id2idx)
edges_tgt = edges_df['node2'].astype(str).map(id2idx)
mask = edges_src.notna() & edges_tgt.notna()
edges_src = edges_src[mask].astype(int).to_numpy()
edges_tgt = edges_tgt[mask].astype(int).to_numpy()

if len(edges_src) == 0:
    raise RuntimeError("❌ No valid edges matched node IDs.")

edge_array = np.vstack([
    np.concatenate([edges_src, edges_tgt]),
    np.concatenate([edges_tgt, edges_src])
]).astype(np.int64)
edge_index = torch.from_numpy(edge_array).long()

# ==========================================================
# 6. NODE FEATURES (Production Stats)
# ==========================================================
prod_numeric = prod_df.drop(columns=['Date'], errors='ignore').apply(pd.to_numeric, errors='coerce').fillna(0)
mean_df = prod_numeric.mean().reset_index().rename(columns={'index': 'Node', 0: 'ProdMean'})
std_df  = prod_numeric.std().reset_index().rename(columns={'index': 'Node', 0: 'ProdStd'})
feat_df = mean_df.merge(std_df, on='Node', how='left').fillna(0)

nodes_feat = nodes_df[['Node']].merge(feat_df, on='Node', how='left').fillna(0)
scaler_x = StandardScaler()
X_scaled = scaler_x.fit_transform(nodes_feat[['ProdMean', 'ProdStd']])
x = torch.tensor(X_scaled, dtype=torch.float32)

# ==========================================================
# 7. TARGET (Average Production)
# ==========================================================
nodes_df = nodes_df.merge(mean_df, on='Node', how='left').fillna(0)
y_raw = nodes_df['ProdMean'].values.astype(float).reshape(-1, 1)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y_raw)
y = torch.tensor(y_scaled, dtype=torch.float32)

print(f"✅ Node features shape: {x.shape} | Target shape: {y.shape}")

# ==========================================================
# 8. BUILD PyG DATA
# ==========================================================
data = Data(x=x, edge_index=edge_index, y=y)
idx = np.arange(num_nodes)
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED)
train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=SEED)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_idx] = True
val_mask[val_idx]     = True
test_mask[test_idx]   = True

data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask

# ==========================================================
# 9. MODEL DEFINITION
# ==========================================================
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(in_channels=data.num_node_features, hidden_channels=64, out_channels=1).to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = torch.nn.MSELoss()

# ==========================================================
# 10. TRAINING LOOP
# ==========================================================
def train_epoch():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def eval_mae_scaled(mask):
    model.eval()
    with torch.no_grad():
        out = model(data).cpu().numpy().flatten()
        y_true = data.y.cpu().numpy().flatten()
    return mean_absolute_error(y_true[mask.cpu().numpy()], out[mask.cpu().numpy()])

EPOCHS = 200
PATIENCE = 30
best_val = float('inf')
best_state = None
wait = 0

print("🚀 Starting training...")
for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch()
    val_mae = eval_mae_scaled(data.val_mask)
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val MAE (scaled): {val_mae:.6f}")
    if val_mae < best_val:
        best_val = val_mae
        best_state = model.state_dict()
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"⏹️ Early stopping at epoch {epoch}")
            break

if best_state is not None:
    model.load_state_dict(best_state)

# ==========================================================
# 11. EVALUATION
# ==========================================================
model.eval()
with torch.no_grad():
    preds_scaled = model(data).cpu().numpy().flatten()
    y_scaled_all = data.y.cpu().numpy().flatten()

preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
actual = scaler_y.inverse_transform(y_scaled_all.reshape(-1, 1)).flatten()

mae = mean_absolute_error(actual, preds)
rmse = np.sqrt(((actual - preds) ** 2).mean())
r2 = r2_score(actual, preds)
print(f"📊 Final Metrics — MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")

# ==========================================================
# 12. SAVE ARTIFACTS & VISUALIZATION
# ==========================================================
os.makedirs("/kaggle/working/artifacts", exist_ok=True)
torch.save(model.state_dict(), "/kaggle/working/artifacts/gcn_model.pt")

comp_df = pd.DataFrame({"Node": nodes_df['Node'], "Actual": actual, "Predicted": preds})
comp_df.to_csv("/kaggle/working/artifacts/predictions_all.csv", index=False)

# PCA plot of embeddings
with torch.no_grad():
    emb = model.conv1(data.x, data.edge_index).cpu().numpy()
pca = PCA(n_components=2)
coords = pca.fit_transform(emb)

plt.figure(figsize=(6, 5))
sc = plt.scatter(coords[:, 0], coords[:, 1], c=actual, cmap="viridis", s=80)
plt.colorbar(sc, label="Average Production (actual)")
plt.title("Node Embeddings Learned by GCN")
plt.xlabel("Embedding 1")
plt.ylabel("Embedding 2")
plt.tight_layout()
plt.savefig("/kaggle/working/artifacts/embeddings_by_production.png", bbox_inches='tight')
plt.show()

print("✅ Model and artifacts saved to /kaggle/working/artifacts/")
