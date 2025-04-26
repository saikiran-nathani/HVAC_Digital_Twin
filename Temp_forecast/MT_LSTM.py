import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
file_path = "hvac_data_cleaned.csv"
df = pd.read_csv(file_path)
data=df.dropna(how='all')

# Selecting relevant features
features = ['field1', 'field2', 'field3', 'field4', 'field5', 'field6', 'field7', 'field8']
data = df[features].dropna()

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Function to create sequences
def create_sequences(data, seq_length=10):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)

seq_length = 10
X, y = create_sequences(data_scaled, seq_length=seq_length)

# Train-test split
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# Custom Dataset
class HVACDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# DataLoader
batch_size = 128
train_dataset = HVACDataset(X_train_tensor, y_train_tensor)
test_dataset = HVACDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Define MT-LSTM Cell
class MT_LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MT_LSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.forget_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.cell_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat((x, h_prev), dim=1)
        forget = self.sigmoid(self.layer_norm(self.forget_gate(combined)))
        input_gate = self.sigmoid(self.layer_norm(self.input_gate(combined)))
        output_gate = self.sigmoid(self.layer_norm(self.output_gate(combined)))
        cell_input = self.tanh(self.layer_norm(self.cell_gate(combined)))
        c_new = (forget * c_prev) + (input_gate * cell_input)
        h_new = output_gate * self.tanh(c_new)
        return h_new, c_new

# Define MT-LSTM Model
class MT_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.3):
        super(MT_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm_layers = nn.ModuleList(
            [MT_LSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = [torch.zeros(batch_size, self.hidden_dim).to(x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_dim).to(x.device) for _ in range(self.num_layers)]

        for t in range(seq_len):
            x_t = x[:, t, :]
            for i, lstm_layer in enumerate(self.lstm_layers):
                h[i], c[i] = lstm_layer(x_t, h[i], c[i])
                x_t = h[i]

        output = self.fc(self.dropout(h[-1]))
        return output

# Initialize Model
input_dim = len(features)
hidden_dim = 512  # Increased hidden size
num_layers = 4
output_dim = len(features)
model = MT_LSTM(input_dim, hidden_dim, num_layers, output_dim).to(device)

# Training Setup
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

def train_model(model, train_loader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step(total_loss)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

def evaluate_model(model, test_loader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    plt.figure(figsize=(12, 8))
    forecast_horizon = 5  # Adjust based on your forecasting needs

    for i, field in enumerate(features, 1):
        plt.subplot(4, 2, i)

        actual_vals = y_test[:, i - 1][-100:]
        forecast_vals = predictions[:forecast_horizon, i - 1]

        plt.plot(actual_vals, label='Actual', color='blue')
        plt.plot(np.arange(len(actual_vals), len(actual_vals) + forecast_horizon), forecast_vals,
                 label='Forecast', color='red', linestyle='dashed')

        # Limit x-axis to last 100 values + forecast horizon
        plt.xlim(0, len(actual_vals) + forecast_horizon)

        # Limit y-axis based on min/max of actual and forecast values with padding
        y_min = min(np.min(actual_vals), np.min(forecast_vals))
        y_max = max(np.max(actual_vals), np.max(forecast_vals))
        padding = (y_max - y_min) * 0.1
        plt.ylim(y_min - padding, y_max + padding)

        plt.title(field)
        plt.legend()

    plt.tight_layout()
    plt.show()

    for i, field in enumerate(features):
        print(f"Processing {field}")
        print(f"Forecast: {predictions[:5, i]}")
        mae = mean_absolute_error(actuals[:, i], predictions[:, i])
        mse = mean_squared_error(actuals[:, i], predictions[:, i])
        r_squared = r2_score(actuals[:, i], predictions[:, i])
        print(f"MAE: {mae:.4f}, MSE: {mse:.4f}")


plt.figure(figsize=(12, 8))
forecast_horizon = 5  # Adjust based on your forecasting needs
# Train and Evaluate
if __name__ == '__main__':
    train_model(model, train_loader, num_epochs=50)
    evaluate_model(model, test_loader)