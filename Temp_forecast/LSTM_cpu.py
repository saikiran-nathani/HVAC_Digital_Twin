import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the dataset
file_path = "thingspeak_all_data.csv"  # Update this path if needed
data = pd.read_csv(file_path)
print("Data loaded successfully!")

# Preprocess the data
data = data.fillna(method='ffill')  # Forward fill to handle missing values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['field1']].values)  # Use 'field1' as the target for now


# Prepare sequences for LSTM
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append((seq, label))
    return sequences


seq_length = 32
sequences = create_sequences(scaled_data, seq_length)

# Convert to PyTorch tensors
X = torch.tensor([s[0] for s in sequences], dtype=torch.float32).to(device)
y = torch.tensor([s[1] for s in sequences], dtype=torch.float32).to(device)

# Create DataLoader
batch_size = 64
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# Initialize the model, loss, and optimizer
model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in dataloader:
        y_batch = y_batch.view(-1, 1)  # Ensure the target shape matches the output

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_loss = train_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

print("Training completed!")
