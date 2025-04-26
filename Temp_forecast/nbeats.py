import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import requests
import time
import io
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
import logging
from datetime import datetime


import warnings
warnings.filterwarnings("ignore")


logging.basicConfig(level=logging.INFO)


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


def clean_and_smooth_data(data, window_size=5):
    if 'created_at' in data.columns:
        data.drop(['entry_id'], axis=1, inplace=True)


    for col in data.columns:
        if col != 'created_at':
            data[col] = preprocess_field(data[col])
            data[col] = data[col].rolling(window=window_size, min_periods=1).mean()

    data.fillna(method='ffill', inplace=True)
    return data


def preprocess_field(series):
    series = series.replace(r'[^0-9.-]', '', regex=True)
    series = pd.to_numeric(series, errors='coerce')
    series.fillna(method='ffill', inplace=True)
    return series


def calculate_metrics(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((actual - forecast) ** 2) / np.sum((actual - np.mean(actual)) ** 2)
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    return mae, mse, rmse, r2, mape


def forecast_nbeats(data, forecast_horizon=10):
    forecast_results = {}
    error_metrics = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for field in data.columns:
        if field == 'created_at':
            continue
        logging.info(f"\nProcessing {field}")

        series = data[field].dropna()
        if len(series) < 30:
            logging.warning(f"Not enough data points for {field}")
            continue


        ts_data = pd.DataFrame({
            'unique_id': ['HVAC'] * len(series),
            'ds': pd.date_range(start='2024-01-01', periods=len(series), freq='T'),
            'y': series.values
        })


        model = NBEATS(h=forecast_horizon, input_size=60, max_steps=5000, learning_rate=1e-4)
        nf = NeuralForecast(models=[model], freq='T')
        nf.fit(ts_data)


        nf.models[0].to(device)


        forecast = nf.predict()
        forecast_values = forecast['NBEATS'].values.flatten()
        forecast_results[field] = forecast_values


        actual = series[-forecast_horizon:]
        if len(actual) == forecast_horizon:
            mae, mse, rmse, r2, mape = calculate_metrics(actual, forecast_values)


            logging.info(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
            error_metrics.append({
                'Field': field,
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R²': r2,
                'MAPE': mape
            })


    num_fields = len([field for field in data.columns if field != 'created_at'])
    num_cols = 2
    num_rows = (num_fields + num_cols - 1) // num_cols
    plt.figure(figsize=(15, num_rows * 5))

    subplot_index = 1
    for field in data.columns:
        if field == 'created_at':
            continue
        if field in forecast_results:
            plt.subplot(num_rows, num_cols, subplot_index)
            series = data[field].dropna()
            plt.plot(series[-100:], label='Actual')
            forecast_index = np.arange(len(series), len(series) + forecast_horizon)
            forecast_series = pd.Series(forecast_results[field], index=forecast_index)
            plt.plot(forecast_series, label='Forecast')
            plt.title(field)
            plt.legend()
            subplot_index += 1

    plt.tight_layout()


    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    forecast_filename = f"forecast_plot_{timestamp}.png"
    plt.savefig(forecast_filename)
    logging.info(f"Forecast plot saved as {forecast_filename}")


    error_metrics_df = pd.DataFrame(error_metrics)
    fig, ax = plt.subplots(figsize=(10, 5))  # Create a new figure for the metrics
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=error_metrics_df.values, colLabels=error_metrics_df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)


    metrics_filename = f"error_metrics_{timestamp}.png"
    plt.savefig(metrics_filename)
    logging.info(f"Error metrics saved as {metrics_filename}")

    return forecast_results


if __name__ == "__main__":
    CHANNEL_ID = '2834542'
    READ_API_KEY = 'SR5O5P9FU4Z93RF9'

    while True:
        try:

            hvac_data = fetch_data_from_thingspeak(CHANNEL_ID, READ_API_KEY)
            if not hvac_data.empty:
                hvac_data = clean_and_smooth_data(hvac_data)


                forecast_results = forecast_nbeats(hvac_data)

            else:
                logging.warning("No data fetched. Retrying in 1 minute...")


            time.sleep(900)

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            time.sleep(60)
