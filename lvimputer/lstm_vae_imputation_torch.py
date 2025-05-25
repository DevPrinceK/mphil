import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Load data (assuming data.txt is downloaded and unzipped from Intel Lab repository)
col_names = ['date','time','epoch','moteid','temperature','humidity','light','voltage']
df = pd.read_csv('data.txt', sep=r'\s+', names=col_names, header=None)

# Select one sensor (e.g., moteid=1) and sort by time
sensor_id = 1
df = df[df.moteid == sensor_id].sort_values(['date','time']).reset_index(drop=True)

# Parse combined datetime and extract temperature series
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
df = df.set_index('datetime')
temperature_series = df['temperature'].astype(float)
print("Loaded {} readings".format(len(temperature_series)))


# Convert series to NumPy array
data = temperature_series.values
N = len(data)

# Choose missingness rate (e.g., 20%)
missing_rate = 0.3
np.random.seed(42)
mask = np.ones(N, dtype=bool)
missing_indices = np.random.choice(N, size=int(N * missing_rate), replace=False)
mask[missing_indices] = False

# Force missing values at the start and end
mask[:5] = False  # first 5 values missing
mask[-5:] = False  # last 5 values missing

# Mask outliers in addition to random missing values
# Using rolling 3-sigma (24-hour window) times 7 days for outlier detection
window = 24  # 24 hours times 7 days, assuming hourly data
rolling_mean = pd.Series(data, index=df.index).rolling(window=window, min_periods=1, center=True).mean().values
rolling_std = pd.Series(data, index=df.index).rolling(window=window, min_periods=1, center=True).std().values
outlier_threshold = 3
outlier_mask = np.abs(data - rolling_mean) > (outlier_threshold * rolling_std)

# Combine with existing mask (missing or outlier)
mask = mask & (~outlier_mask)

# Create observed series with NaNs for missing entries
observed = data.copy()
observed[~mask] = 0.0   # fill missing and outliers with 0 (or any placeholder)

# Prepare input for model: shape (seq_len, 1) for value and (seq_len, 1) for mask
input_values = torch.tensor(observed, dtype=torch.float32).unsqueeze(1)  # shape [N,1]
input_mask   = torch.tensor(mask.astype(float), dtype=torch.float32).unsqueeze(1)  # shape [N,1]
# Concatenate to shape [N, 2]
model_input  = torch.cat([input_values, input_mask], dim=1)

class LSTMVAE(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, latent_dim=16, num_layers=1): # testing with 1 layers
        """
        LSTM Variational Autoencoder for time series imputation.
        Args:
            input_dim (int): Number of input features (e.g., 2 for value and mask).
            hidden_dim (int): Number of LSTM hidden units.
            latent_dim (int): Dimensionality of the latent space.
            num_layers (int): Number of LSTM layers.
        """
        super().__init__()
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Latent space
        self.fc_mu  = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder initialization
        self.fc_decode = nn.Linear(latent_dim, hidden_dim)
        # Decoder LSTM and output layer
        self.decoder_lstm = nn.LSTM(1, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 1)  # reconstruct single value

    def encode(self, x):
        # x: [batch, seq_len, input_dim]
        _, (h_n, _) = self.encoder_lstm(x)
        h = h_n[-1]  # take last layer's hidden state
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        num_layers = self.decoder_lstm.num_layers
        batch_size = z.size(0)
        hidden_dim = self.decoder_lstm.hidden_size
        # Prepare latent to hidden and initial input
        h_dec_single = torch.tanh(self.fc_decode(z)).unsqueeze(0)  # [1, batch, hidden_dim]
        h_dec = h_dec_single.repeat(num_layers, 1, 1)  # [num_layers, batch, hidden_dim]
        c_dec = torch.zeros_like(h_dec)
        inp = torch.zeros(batch_size, seq_len, 1).to(z.device)
        dec_out, _ = self.decoder_lstm(inp, (h_dec, c_dec))
        recon = self.fc_out(dec_out).squeeze(-1)  # [batch, seq_len]
        return recon

    def forward(self, x):
        # x: [batch, seq_len, 2]
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        seq_len = x.size(1)
        recon_seq = self.decode(z, seq_len)
        # Ensure output shape is [batch, seq_len, 1] for consistency
        return recon_seq.unsqueeze(-1), mu, logvar

# Define loss function
def vae_loss(recon_x, x_true, mask, mu, logvar):
    # Ensure all tensors are the same shape
    if recon_x.shape != x_true.shape:
        recon_x = recon_x.view_as(x_true)
    # Compute reconstruction loss (MSE) only on observed data
    mse = ((recon_x - x_true) ** 2)
    eps = 1e-8
    mse = (mse * mask).sum() / (mask.sum() + eps)  # average over observed points
    # KL Divergence between q(z|x) and N(0,I)
    batch_size = mu.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    return mse + kld

# Prepare data (as batch of one sequence for simplicity)
x_train = model_input.unsqueeze(0)  # [batch=1, seq_len, 2]
y_train = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # ground truth [batch, seq_len, 1]
mask_train = torch.tensor(mask.astype(float), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

# Device handling
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMVAE(input_dim=2, hidden_dim=64, latent_dim=16).to(device)
x_train = x_train.to(device)
y_train = y_train.to(device)
mask_train = mask_train.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(300):  # e.g., 300 epochs
    optimizer.zero_grad()
    recon_seq, mu, logvar = model(x_train)
    loss = vae_loss(recon_seq, y_train, mask_train, mu, logvar)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    recon_seq, _, _ = model(x_train)  # reconstruct full sequence
    recon = recon_seq.squeeze().cpu().numpy()
    # Ensure recon shape matches data for correct indexing
    if recon.shape != data.shape:
        recon = recon.reshape(data.shape)
    
# Compute metrics only on missing positions
true_vals = data[~mask]
imputed_vals = recon[~mask]
mae = np.mean(np.abs(imputed_vals - true_vals))
rmse = np.sqrt(np.mean((imputed_vals - true_vals) ** 2))
print(f"Imputation MAE: {mae:.4f}, RMSE: {rmse:.4f}")
# Save model and optimizer state
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, 'lstm_vae_model.pth')
