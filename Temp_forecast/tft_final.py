import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
import os
import requests
import time
import io
import warnings

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def fetch_data_from_thingspeak(channel_id, read_api_key, results=8000):
    url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.csv?api_key={read_api_key}&results={results}"
    response = requests.get(url)
    data = pd.read_csv(io.StringIO(response.text))
    data['created_at'] = pd.to_datetime(data['created_at'])
    return data


def clean_data(data):
    if 'created_at' in data.columns:
        data.drop(['entry_id'], axis=1, inplace=True)
    for col in data.columns:
        data[col] = preprocess_field(data[col])
    data.fillna(method='ffill', inplace=True)
    return data


def preprocess_field(series):
    series = series.replace(r'[^0-9.-]', '', regex=True)
    series = pd.to_numeric(series, errors='coerce')
    series.fillna(method='ffill', inplace=True)
    return series


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


class TemporalFusionTransformer(nn.Module):
    def __init__(self, seq_length, n_features, num_heads=4, hidden_units=128, dropout_rate=0.2):
        super(TemporalFusionTransformer, self).__init__()
        self.seq_length = seq_length
        self.n_features = n_features
        self.hidden_units = hidden_units

        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, hidden_units))
        self.feature_proj = nn.Linear(n_features, hidden_units)

        self.attn_layer1 = nn.MultiheadAttention(embed_dim=hidden_units, num_heads=num_heads, dropout=dropout_rate)
        self.attn_layer2 = nn.MultiheadAttention(embed_dim=hidden_units, num_heads=num_heads, dropout=dropout_rate)

        self.feedforward = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Added dropout
            nn.Linear(hidden_units, hidden_units),
            nn.Dropout(dropout_rate)   # Added dropout
        )

        self.output_layer = nn.Linear(hidden_units, 1)
        self.layer_norm1 = nn.LayerNorm(hidden_units)
        self.layer_norm2 = nn.LayerNorm(hidden_units)

    def forward(self, x):
        x = self.feature_proj(x) + self.positional_encoding

        attn_output1, _ = self.attn_layer1(x, x, x)
        x = self.layer_norm1(attn_output1 + x)

        attn_output2, _ = self.attn_layer2(x, x, x)
        x = self.layer_norm2(attn_output2 + x)

        x = self.feedforward(x)
        return self.output_layer(x)


def train_model(model, train_loader, val_loader, seq_length, device, n_epochs=100, lr=0.0005):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
    criterion = nn.HuberLoss()

    best_val_loss = float('inf')
    model.train()

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch).squeeze(-1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze(-1)
                val_loss += criterion(outputs, y_batch).item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print(f"Saved best model with val loss: {val_loss:.4f}")

    model.load_state_dict(torch.load("best_model.pt"))
    return model


def calculate_metrics(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    r2 = np.corrcoef(actual, forecast)[0, 1] ** 2
    return mae, mse, rmse, r2


CHANNEL_ID = '2834542'
READ_API_KEY = 'SR5O5P9FU4Z93RF9'
SEQ_LENGTH = 30
BATCH_SIZE = 64
N_EPOCHS = 100
LEARNING_RATE = 0.0005


def build_tft_model(seq_length, n_features):
    model = TemporalFusionTransformer(seq_length, n_features).to(device)
    return model


while True:
    try:
        print("Fetching data from ThingSpeak...")
        hvac_data = fetch_data_from_thingspeak(CHANNEL_ID, READ_API_KEY)
        hvac_data_cleaned = clean_data(hvac_data)

        forecast_results = {}  # Store forecast results for each field
        metrics_data = []  # Store metrics for each field

        for field in hvac_data_cleaned.columns:
            if field == "created_at":
                continue

            print(f"Processing field: {field}")
            series = hvac_data_cleaned[field].values.reshape(-1, 1)

            scaler = MinMaxScaler()
            series_scaled = scaler.fit_transform(series)

            X, y = create_sequences(series_scaled, SEQ_LENGTH)

            train_size = int(0.8 * len(X))
            X_train, y_train = X[:train_size], y[:train_size]
            X_val, y_val = X[train_size:], y[train_size:]

            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                          torch.tensor(y_train, dtype=torch.float32))
            val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                        torch.tensor(y_val, dtype=torch.float32))

            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

            model = build_tft_model(SEQ_LENGTH, n_features=1)

            print(f"Training model for field: {field}")
            model = train_model(model, train_loader, val_loader, SEQ_LENGTH, device,
                                n_epochs=N_EPOCHS, lr=LEARNING_RATE)

            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
                y_pred = model(X_val_tensor).cpu().numpy()

            y_pred = y_pred[:, -1, :].flatten()

            y_val_reshaped = y_val.reshape(-1, 1)

            y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            y_val_inverse = scaler.inverse_transform(y_val_reshaped).flatten()

            mae, mse, rmse, r2 = calculate_metrics(y_val_inverse, y_pred)
            print(f"Metrics for {field} -> MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")

            metrics_data.append([field, mae, mse, rmse, r2])

            # Store forecast results for plotting
            forecast_results[field] = y_pred[-10:]

        # Plot the forecast results
        num_fields = len([field for field in hvac_data.columns if field != 'created_at'])
        num_cols = 2
        num_rows = (num_fields + num_cols - 1) // num_cols
        plt.figure(figsize=(15, num_rows * 5))

        subplot_index = 1
        for field in hvac_data.columns:
            if field == 'created_at':
                continue
            if field in forecast_results:
                plt.subplot(num_rows, num_cols, subplot_index)
                plt.plot(hvac_data[field][-100:], label='Actual')
                forecast_index = np.arange(len(hvac_data[field]), len(hvac_data[field]) + 10)
                forecast_series = pd.Series(forecast_results[field], index=forecast_index)
                plt.plot(forecast_series, label='Forecast')
                plt.title(field)
                plt.legend()
                subplot_index += 1

        plt.tight_layout()

        # Generate a timestamp for filenames
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Save the prediction plot
        plot_filename = f"tft_predictions_{timestamp}.png"
        plt.savefig(plot_filename)
        print(f"Prediction plot saved as {plot_filename}")

        # Save metrics as a table
        metrics_df = pd.DataFrame(metrics_data, columns=["Field", "MAE", "MSE", "RMSE", "R2"])
        metrics_df.set_index("Field", inplace=True)

        # Plot metrics as a table and save
        plt.figure(figsize=(10, len(metrics_data) * 0.6))
        plt.axis('off')
        table = plt.table(cellText=metrics_df.values,
                          colLabels=metrics_df.columns,
                          rowLabels=metrics_df.index,
                          cellLoc='center',
                          loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(metrics_df.columns))))

        metrics_filename = f"tft_metrics_{timestamp}.png"
        plt.savefig(metrics_filename, bbox_inches='tight')
        print(f"Metrics saved as {metrics_filename}")

        print(forecast_results)

    except Exception as e:
        print(f"Error occurred: {e}")
        time.sleep(60)