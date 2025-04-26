import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the data
hvac_data = pd.read_csv('hvac_data.csv')
print(hvac_data.head())

# Check for missing values
missing_values = hvac_data.isnull().sum()
print("Missing values:\n", missing_values)

# Fill missing values using forward fill
hvac_data.fillna(method='ffill', inplace=True)

# Function to prepare data for LSTM
def prepare_data_for_lstm(data, seq_length):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)
    sequences = []
    for i in range(len(scaled_data) - seq_length):
        sequences.append(scaled_data[i:i + seq_length])
    return np.array(sequences), scaler

SEQ_LENGTH = 10  # Define the sequence length for LSTM (adjust as needed)

# Prepare data for LSTM
X, scaler = prepare_data_for_lstm(hvac_data.iloc[:, 2:10], SEQ_LENGTH)
y = scaler.transform(hvac_data.iloc[:, 2:10].values[SEQ_LENGTH:])  # Assuming we are forecasting the next step

# Reshape X to be 3-dimensional [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(units=8))  # Assuming 8 output features (adjust as needed)

# Compile the model
model.compile(optimizer='adam', loss='mse')  # Mean Squared Error (MSE) loss for regression tasks

# Fit the model
model.fit(X, y, epochs=50, batch_size=32, verbose=1)  # Adjust epochs and batch_size as needed

# Function to make forecasts using trained LSTM model
def make_forecasts(model, data, scaler, seq_length, forecast_steps):
    forecasts = []
    for i in range(len(data) - seq_length + 1):
        if i >= len(data) - forecast_steps:
            break
        seq = data[i:i + seq_length]
        seq = seq.reshape((1, seq.shape[0], seq.shape[1]))
        forecast = model.predict(seq)
        forecasts.append(forecast)
    forecasts = np.array(forecasts)
    forecasts = forecasts.reshape((forecasts.shape[0], forecasts.shape[2]))
    forecasts = scaler.inverse_transform(forecasts)  # Inverse transform forecasts to original scale
    return forecasts

# Make forecasts
forecast_steps = 5  # Number of steps to forecast
forecasts = make_forecasts(model, hvac_data.iloc[:, 2:10].values, scaler, SEQ_LENGTH, forecast_steps)

print("Forecasts shape:", forecasts.shape)
print("Forecasts:\n", forecasts)

import matplotlib.pyplot as plt

# Plot the results
plt.figure(figsize=(12, 8))
for i, field in enumerate(hvac_data.columns[2:10], 1):
    plt.subplot(4, 2, i)
    plt.plot(hvac_data[field].values[-100:], label='Actual')
    plt.plot(np.arange(len(hvac_data[field].values)-5, len(hvac_data[field].values)), forecasts[:, i-1], label='Forecast')
    plt.title(field)
    plt.legend()
plt.tight_layout()
plt.show()

