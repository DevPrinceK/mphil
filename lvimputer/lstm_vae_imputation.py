
# LSTM + VAE Imputation for IoT Sensor Data (Intel Lab)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models

# Load Intel Lab dataset
col_names = ['date', 'time', 'epoch', 'moteid', 'temperature', 'humidity', 'light', 'voltage']
df = pd.read_csv('data.txt', sep='\s+', names=col_names, header=None)

# Filter for moteid 1
df = df[df['moteid'] == 1].sort_values(['date', 'time']).reset_index(drop=True)
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
df = df.set_index('datetime')
temperature_series = df['temperature'].astype(float)

# Normalize data
scaler = MinMaxScaler()
temperature_scaled = scaler.fit_transform(temperature_series.values.reshape(-1, 1)).flatten()

# Simulate missing data
np.random.seed(42)
N = len(temperature_scaled)
missing_rate = 0.2
mask = np.ones(N, dtype=bool)
missing_indices = np.random.choice(N, int(N * missing_rate), replace=False)
mask[missing_indices] = False

data = temperature_scaled
observed = data.copy()
observed[~mask] = 0.0

# Prepare model input
values = observed.reshape(-1, 1)
masks = mask.astype(float).reshape(-1, 1)
inputs = np.concatenate([values, masks], axis=1)
inputs = np.expand_dims(inputs, axis=0)
ground_truth = np.expand_dims(data, axis=(0, -1))

# Build VAE-LSTM
timesteps = inputs.shape[1]
latent_dim = 16

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vae_lstm(timesteps, latent_dim):
    encoder_inputs = layers.Input(shape=(timesteps, 2))
    x = layers.LSTM(64)(encoder_inputs)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    z = Sampling()([z_mean, z_log_var])

    decoder_input = layers.Dense(timesteps * 64, activation='tanh')(z)
    decoder_input = layers.Reshape((timesteps, 64))(decoder_input)
    x_decoded = layers.LSTM(64, return_sequences=True)(decoder_input)
    outputs = layers.TimeDistributed(layers.Dense(1))(x_decoded)

    vae = models.Model(encoder_inputs, outputs)
    kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
    return vae, kl_loss

vae, kl_loss = build_vae_lstm(timesteps, latent_dim)

def vae_loss_fn(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square((y_true - y_pred) * masks.reshape(1, -1, 1)))
    return mse_loss + kl_loss

vae.compile(optimizer='adam', loss=vae_loss_fn)

# Train model
vae.fit(inputs, ground_truth, epochs=100, verbose=0)

# Predict and evaluate
recon = vae.predict(inputs)[0].squeeze()

true_vals = data[~mask]
imputed_vals = recon[~mask]
mae = np.mean(np.abs(imputed_vals - true_vals))
rmse = np.sqrt(np.mean((imputed_vals - true_vals)**2))
print(f"Imputation MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# Plot
plt.figure(figsize=(12, 5))
plt.plot(data, label='Original', alpha=0.6)
plt.plot(np.where(mask, np.nan, recon), 'ro', label='Imputed')
plt.title("LSTM-VAE Imputation (TensorFlow)")
plt.legend()
plt.show()
