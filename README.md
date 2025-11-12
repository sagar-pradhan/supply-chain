# 🧠 Supply Chain Graph Neural Network (GCN)

A complete implementation of a **Graph Neural Network (GCN)** for supply chain production prediction using **PyTorch Geometric**.  
Demonstrates practical graph modeling, feature scaling, and embedding visualization — all in a single Kaggle cell.

---

## 🚀 Overview

Supply chains form naturally connected systems — plants, suppliers, and distributors interact as **graph nodes**, connected by **edges** that represent flow of materials or information.  
This project applies a **Graph Neural Network (GCN)** to model these relationships and predict each node’s average production.

It is **fully Kaggle-ready** — you can copy one cell of code into Kaggle, attach the dataset, and train instantly.

---

## 🧩 Pipeline Summary

1. **Automatic Dependency Installation**  
   - Installs `torch-geometric` and dependencies if missing.  
   - Compatible with **PyTorch 2.6.0 + CUDA 12.4** (Kaggle default).

2. **Data Loading**  
   - Loads from:
     - `Nodes.csv`
     - `Edges (Plant).csv`
     - `Production .csv`
   - Builds a graph where nodes = plants and edges = supply relations.

3. **Feature Engineering**  
   - Computes mean and standard deviation of production per node.  
   - Standardizes all numerical features.

4. **Graph Construction**  
   - Creates a `torch_geometric.data.Data` object with node features, edge connections, and train/val/test masks.

5. **GCN Model**  
   - Two-layer GCN with ReLU + dropout.
   - Learns node embeddings and predicts average production.

6. **Training & Evaluation**  
   - Uses MSE loss, Adam optimizer, and early stopping based on validation MAE.  
   - Evaluates using MAE, RMSE, and R² metrics.

7. **Visualization & Saving**  
   - Uses PCA to project learned node embeddings.  
   - Saves results to `/kaggle/working/artifacts/`.

---

## 📂 Dataset Structure
/kaggle/input/supplygraph-supply-chain-planning-using-gnns/Raw Dataset/
Raw Dataset/
├── Nodes/
│ └── Nodes.csv
├── Edges/
│ └── Edges (Plant).csv
└── Temporal Data/
└── Unit/
└── Production .csv


### CSV Format Expectations

**Nodes.csv**
| Column | Description |
|--------|--------------|
| Node | Unique plant or unit identifier |

**Edges (Plant).csv**
| Column | Description |
|--------|--------------|
| node1 | Source node |
| node2 | Target node |

**Production .csv**
| Column | Description |
|--------|--------------|
| Date | Timestamp (optional) |
| Node1, Node2, ... | Production values per node |

---

## 🧑‍💻 How to Run (Kaggle Instructions)

1. Go to **[Kaggle → Notebooks](https://www.kaggle.com/code)**.  
2. Create a **new notebook**.  
3. In the **first cell**, paste the full GCN code (from this repo).  
4. Click “**Add Data**” → attach the dataset:  
   - Search for **“supplygraph-supply-chain-planning-using-gnns”**.  
5. Run all cells — installation, data loading, model training, and saving will happen automatically.  
6. Check `/kaggle/working/artifacts/` for:
   - `gcn_model.pt`
   - `predictions_all.csv`
   - `embeddings_by_production.png`

---


