import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load and preprocess HVAC data
hvac_data = pd.read_csv('hvac_data.csv')
hvac_data.fillna(method='ffill', inplace=True)
scaler = MinMaxScaler()
hvac_scaled = scaler.fit_transform(hvac_data.iloc[:, 1:])

# Create Graph Structure for GCN
num_nodes = hvac_scaled.shape[1]
edge_index = torch.tensor([(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j], dtype=torch.long).t().contiguous()

class GraphTemporalFusionTransformer(nn.Module):
    def __init__(self, seq_length, n_features, hidden_units=128, num_heads=4):
        super(GraphTemporalFusionTransformer, self).__init__()
        self.seq_length = seq_length
        self.n_features = n_features

        # Graph Convolutional Layer
        self.gcn = pyg_nn.GCNConv(n_features, hidden_units)

        # Dynamic Feature Embedding
        self.feature_embedding = nn.Linear(hidden_units, hidden_units)

        # Causal Temporal Attention
        self.attn_layer = nn.MultiheadAttention(embed_dim=hidden_units, num_heads=num_heads)

        # Fully Connected Layers
        self.fc = nn.Linear(hidden_units, 1)
        self.layer_norm = nn.LayerNorm(hidden_units)

    def forward(self, x, edge_index):
        # Pass through GCN
        x = self.gcn(x, edge_index)
        x = self.feature_embedding(x)

        # Apply Causal Temporal Attention
        x = x.unsqueeze(1).repeat(1, self.seq_length, 1)
        attn_output, _ = self.attn_layer(x, x, x)
        x = self.layer_norm(attn_output + x)

        # Final prediction
        return self.fc(x[:, -1, :])

# Prepare time series sequences
SEQ_LENGTH = 30
X, y = [], []
for i in range(len(hvac_scaled) - SEQ_LENGTH):
    X.append(hvac_scaled[i:i+SEQ_LENGTH])
    y.append(hvac_scaled[i+SEQ_LENGTH])
X, y = np.array(X), np.array(y)

dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphTemporalFusionTransformer(seq_length=SEQ_LENGTH, n_features=num_nodes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train the model
for epoch in range(20):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        graph_data = Data(x=X_batch.mean(dim=1), edge_index=edge_index.to(device))
        outputs = model(graph_data.x, graph_data.edge_index)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = sum(criterion(model(Data(x=X_val.mean(dim=1).to(device), edge_index=edge_index.to(device)).x, edge_index.to(device)), y_val.to(device)).item() for X_val, y_val in val_loader) / len(val_loader)
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")

# Save model
torch.save(model.state_dict(), 'graph_aware_tft.pt')

# Final Predictions
with torch.no_grad():
    X_test = torch.tensor(hvac_scaled[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, -1), dtype=torch.float32).to(device)
    graph_test = Data(x=X_test.mean(dim=1), edge_index=edge_index.to(device))
    y_pred = model(graph_test.x, graph_test.edge_index).cpu().numpy()

# Reverse Scaling and Plot Results
y_pred = scaler.inverse_transform(y_pred.reshape(1, -1)).flatten()
plt.plot(hvac_data.iloc[-50:, 1], label='Actual')
plt.plot(range(len(hvac_data) - 1, len(hvac_data)), y_pred, label='Graph-Aware TFT Forecast', linestyle='dashed')
plt.legend()
plt.show()
