import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


hvac_data = pd.read_csv('hvac_data_cleaned.csv')


if 'created_at' in hvac_data.columns:
    hvac_data.drop(['created_at', 'entry_id'], axis=1, inplace=True)


for column in hvac_data.columns:
    hvac_data[column] = hvac_data[column].clip(lower=hvac_data[column].quantile(0.05),
                                               upper=hvac_data[column].quantile(0.95))


hvac_data.ffill(inplace=True)


scalers = {}
scaled_data = {}
for column in hvac_data.columns:
    scaler = MinMaxScaler()
    scaled_data[column] = scaler.fit_transform(hvac_data[column].values.reshape(-1, 1))
    scalers[column] = scaler


def create_sequences(data, seq_length=20, forecast_horizon=5):
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + forecast_horizon])  
    return np.array(X), np.array(y)


class PSD_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(PSD_LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        phase_shifted_output, _ = self.lstm1(x)
        dual_state_output, _ = self.lstm2(phase_shifted_output)
        out = self.fc(dual_state_output[:, -1, :])
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


forecast_results = {}
seq_length = 20
forecast_horizon = 5
epochs = 150
lr = 0.001

for field in hvac_data.columns:
    print(f"\nProcessing {field}")
    
    
    data = scaled_data[field]
    X, y = create_sequences(data, seq_length, forecast_horizon)

    
    X_train = torch.tensor(X, dtype=torch.float32).to(device)
    y_train = torch.tensor(y, dtype=torch.float32).squeeze(-1).to(device)  

    
    model = PSD_LSTM(input_size=1, hidden_size=64, output_size=forecast_horizon).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    
    test_input = torch.tensor(data[-seq_length:].reshape(1, seq_length, 1), dtype=torch.float32).to(device)
    forecast = model(test_input).cpu().detach().numpy().flatten()

    
    forecast = scalers[field].inverse_transform(forecast.reshape(-1, 1)).flatten()
    forecast_results[field] = forecast

    
    actual_values = hvac_data[field].values[-forecast_horizon:]  
    mae = mean_absolute_error(actual_values, forecast)
    mse = mean_squared_error(actual_values, forecast)

    
    print(f"Forecast: {forecast}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")


plt.figure(figsize=(12, 8))
for i, field in enumerate(hvac_data.columns, 1):
    if field in forecast_results:
        plt.subplot(4, 2, i)
        plt.plot(hvac_data[field][-100:], label='Actual', color='blue')
        forecast_index = np.arange(len(hvac_data[field]), len(hvac_data[field]) + forecast_horizon)
        plt.plot(forecast_index, forecast_results[field], label='Forecast', color='red', linestyle='dashed')
        plt.title(field)
        plt.legend()
plt.tight_layout()
plt.show()

print(forecast_results)