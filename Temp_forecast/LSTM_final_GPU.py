import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import logging
import time
import io
from sklearn.preprocessing import MinMaxScaler

def fetch_data_from_thingspeak(channel_id, read_api_key, results=8000, retries=5):
    url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.csv?api_key={read_api_key}&results={results}"
    for attempt in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = pd.read_csv(io.StringIO(response.text))
                data['created_at'] = pd.to_datetime(data['created_at'])
                return data
            else:
                logging.warning(f"Failed to fetch data (attempt {attempt + 1}/{retries}). HTTP Status Code: {response.status_code}")
                time.sleep(2 ** attempt)  # Exponential backoff
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {e}")
            time.sleep(2 ** attempt)
    logging.error("Exceeded max retries. Returning empty dataframe.")
    return pd.DataFrame()

def preprocess_field(series):
    series = series.replace(r'[^0-9.-]', '', regex=True)
    series = pd.to_numeric(series, errors='coerce')
    series.fillna(method='ffill', inplace=True)
    return series

# Function to clean and smooth data (reduce noise)
def clean_and_smooth_data(data, window_size=5):
    if 'created_at' in data.columns:
        data.drop(['entry_id'], axis=1, inplace=True)

    # Apply smoothing (moving average) to all numeric columns
    for col in data.columns:
        if col != 'created_at':
            data[col] = preprocess_field(data[col])
            data[col] = data[col].rolling(window=window_size, min_periods=1).mean()  # Smoothing

    data.fillna(method='ffill', inplace=True)
    return data
# Define the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=512, num_layers=4, output_size=10):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Take last timestep output
        return out.unsqueeze(-1)

# Function to preprocess data for LSTM
def prepare_lstm_data(data, input_size=60, forecast_horizon=10):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))

    X, y = [], []
    for i in range(len(data_scaled) - input_size - forecast_horizon):
        X.append(data_scaled[i : i + input_size])
        y.append(data_scaled[i + input_size : i + input_size + forecast_horizon])

    X, y = np.array(X), np.array(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), scaler

# Function to train the LSTM model
def train_lstm(model, train_loader, epochs=100, learning_rate=5e-4, device="cuda"):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# Function to forecast using LSTM
def forecast_lstm(model, data, input_size=60, forecast_horizon=10, device="cuda"):
    model.to(device)
    model.eval()

    X_test, _, scaler = prepare_lstm_data(data, input_size, forecast_horizon)
    X_test = X_test[-1].unsqueeze(0).to(device)  # Take last input sequence

    with torch.no_grad():
        forecast_scaled = model(X_test).cpu().numpy().flatten()
        forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()

    return forecast

# Fetch, clean, and process data
CHANNEL_ID = '2834542'  # Replace with your ThingSpeak channel ID
READ_API_KEY = 'SR5O5P9FU4Z93RF9'  # Replace with your ThingSpeak Read API Key
hvac_data = fetch_data_from_thingspeak(CHANNEL_ID, READ_API_KEY)

if not hvac_data.empty:
    hvac_data = clean_and_smooth_data(hvac_data)
    field = 'field1'  # Change as needed

    # Convert data for LSTM
    series = hvac_data[field].dropna().values
    X, y, scaler = prepare_lstm_data(series)

    # Load data into PyTorch DataLoader
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    # Initialize and train LSTM
    lstm_model = LSTMModel()
    train_lstm(lstm_model, train_loader)

    # Forecast
    forecast_results = forecast_lstm(lstm_model, series)

    # Plot forecast vs actual
    plt.figure(figsize=(10, 5))
    plt.plot(series[-100:], label="Actual")
    plt.plot(np.arange(len(series), len(series) + len(forecast_results)), forecast_results, label="Forecast")
    plt.legend()
    plt.title(f"LSTM Forecast for {field}")
    plt.show()
