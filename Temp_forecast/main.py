import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("merged_data.csv")  # Replace with your actual filename

# Drop non-numeric columns if they are unnecessary
data = data.drop(columns=["created_at"], errors="ignore")  # Drop only if it exists

# Convert all columns to numeric, forcing errors to NaN
data = data.apply(pd.to_numeric, errors='coerce')

# Drop only fully NaN rows, but keep partial data
data = data.dropna(how="all")

# Print to check if data is still valid
print(f"Dataset shape after cleaning: {data.shape}")

# Ensure there is data left
if data.shape[0] == 0:
    raise ValueError("No valid data left after cleaning! Check your dataset.")

# Apply StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

print("Data successfully scaled!")
