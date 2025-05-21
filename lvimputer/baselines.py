import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt

# Load data and mask (assume same as in lstm_vae_imputation_touch.py)
col_names = ['date','time','epoch','moteid','temperature','humidity','light','voltage']
df = pd.read_csv('data.txt', sep=r'\s+', names=col_names, header=None)
sensor_id = 1
df = df[df.moteid == sensor_id].sort_values(['date','time']).reset_index(drop=True)
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
df = df.set_index('datetime')
temperature_series = df['temperature'].astype(float)
data = temperature_series.values
N = len(data)

# Plot the temperature time series
# dates = df.index
# plt.figure(figsize=(15, 5))
# plt.plot(dates, data, label='Temperature')
# plt.xlabel('Datetime')
# plt.ylabel('Temperature')
# plt.title('Temperature Time Series (Sensor {})'.format(sensor_id))
# plt.legend()
# plt.tight_layout()
# plt.show()

# Use the same random mask as in the VAE script for fair comparison
missing_rate = 0.2
np.random.seed(42)
mask = np.ones(N, dtype=bool)
missing_indices = np.random.choice(N, size=int(N * missing_rate), replace=False)
mask[missing_indices] = False

# Force missing values at the start and end
mask[:5] = False  # first 5 values missing
mask[-5:] = False  # last 5 values missing

# Baseline 1: Mean Imputation
mean_value = data[mask].mean()
mean_imputed = data.copy()
mean_imputed[~mask] = mean_value
mean_mae = np.mean(np.abs(mean_imputed[~mask] - data[~mask]))
mean_rmse = np.sqrt(np.mean((mean_imputed[~mask] - data[~mask]) ** 2))

# Baseline 1b: Median Imputation
median_value = np.median(data[mask])
median_imputed = data.copy()
median_imputed[~mask] = median_value
median_mae = np.mean(np.abs(median_imputed[~mask] - data[~mask]))
median_rmse = np.sqrt(np.mean((median_imputed[~mask] - data[~mask]) ** 2))

# Baseline 1c: Mode Imputation
# For continuous data, mode may not be meaningful, but we can use scipy.stats.mode
mode_value = mode(data[mask], keepdims=True).mode[0]
mode_imputed = data.copy()
mode_imputed[~mask] = mode_value
mode_mae = np.mean(np.abs(mode_imputed[~mask] - data[~mask]))
mode_rmse = np.sqrt(np.mean((mode_imputed[~mask] - data[~mask]) ** 2))

# Baseline 2: Forward Fill
ffill_imputed = data.copy()
ffill_imputed[~mask] = np.nan
ffill_imputed = pd.Series(ffill_imputed).ffill().bfill().values  # Use ffill then bfill
ffill_mae = np.mean(np.abs(ffill_imputed[~mask] - data[~mask]))
ffill_rmse = np.sqrt(np.mean((ffill_imputed[~mask] - data[~mask]) ** 2))

# Baseline 3: Linear Interpolation
interp_imputed = data.copy()
interp_imputed[~mask] = np.nan
interp_imputed = pd.Series(interp_imputed).interpolate(method='linear').bfill().ffill().values  # Interpolate, then fill edges
interp_mae = np.mean(np.abs(interp_imputed[~mask] - data[~mask]))
interp_rmse = np.sqrt(np.mean((interp_imputed[~mask] - data[~mask]) ** 2))

print(f"Mean Imputation MAE: {mean_mae:.4f}, RMSE: {mean_rmse:.4f}")
print(f"Median Imputation MAE: {median_mae:.4f}, RMSE: {median_rmse:.4f}")
print(f"Mode Imputation MAE: {mode_mae:.4f}, RMSE: {mode_rmse:.4f}")
print(f"Forward Fill MAE: {ffill_mae:.4f}, RMSE: {ffill_rmse:.4f}")
print(f"Linear Interpolation MAE: {interp_mae:.4f}, RMSE: {interp_rmse:.4f}")
