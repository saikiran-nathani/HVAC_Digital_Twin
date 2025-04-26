import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.fftpack import fft
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from deap import base, creator, tools, algorithms
import random
import warnings

warnings.filterwarnings("ignore")


def fourier_transform(series, num_components=5):
    fft_values = fft(series)
    return np.real(fft_values[:num_components])


# Load the data
hvac_data = pd.read_csv('hvac_data.csv')
hvac_data.fillna(method='ffill', inplace=True)

# Print column names to verify structure
print("Dataset Columns:", hvac_data.columns)

# Check if 'Temperature' column exists
if 'Temperature' not in hvac_data.columns:
    raise KeyError("The 'Temperature' column is missing in the dataset. Please verify column names.")


def arima_forecast(series, order):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit.forecast(steps=5), model_fit.resid


# Define Transformer-based Residual Correction
class ResidualTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(ResidualTransformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc = nn.Linear(hidden_dim, 1)
        self.input_proj = nn.Linear(1, hidden_dim)  # Ensure input is mapped to hidden_dim

    def forward(self, x):
        x = x.unsqueeze(-1)  # Ensure shape is (batch_size, seq_length, 1)
        x = self.input_proj(x)  # Map input to hidden_dim
        x = x.permute(1, 0, 2)  # Transformer expects (seq_length, batch, hidden_dim)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # Convert back to (batch_size, seq_length, hidden_dim)
        return self.fc(x[:, -1, :])  # Take last time step for prediction


# Function to calculate evaluation metrics
def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return mae, mse, rmse


# Instantiate and Train Transformer Model
seq_length = 10
model = ResidualTransformer(input_dim=seq_length, hidden_dim=64)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_size = int(0.8 * len(residual_series) - seq_length)
X_train = torch.tensor(residual_series[:train_size], dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(residual_series[seq_length:train_size + seq_length], dtype=torch.float32)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Train Model
for epoch in range(10):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

# Apply Transformer Correction
with torch.no_grad():
    input_tensor = torch.tensor(residual_series[-seq_length:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    residual_correction = model(input_tensor).item()
final_forecast = forecast + residual_correction

# Compute and print evaluation metrics
mae, mse, rmse = calculate_metrics(hvac_data.iloc[-5:, 1], final_forecast)
print(f"Evaluation Metrics:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# Plot Results (Temperature vs Readings)
plt.figure(figsize=(12, 6))
plt.plot(hvac_data['Temperature'].iloc[-50:], hvac_data.iloc[-50:, 1], label='Actual')
plt.plot(hvac_data['Temperature'].iloc[-5:], forecast, label='ARIMA Forecast')
plt.plot(hvac_data['Temperature'].iloc[-5:], final_forecast, label='Corrected Forecast', linestyle='dashed')
plt.xlabel('Temperature')
plt.ylabel('Readings')
plt.legend()
plt.show()
