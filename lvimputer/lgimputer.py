'''
LGImputer: A Python library for imputing missing values in time series data using LSTM and GRU models.
This library provides a simple interface for training and using LSTM and GRU models to predict missing values in time series data.
'''
import warnings
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# Suppress warnings from PyTorch
# This is useful to avoid cluttering the output with warnings that are not relevant to the user
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Define the dataset class for time series data
class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_length: int):
        self.data = data
        self.seq_length = seq_length

    # Get the length of the dataset
    # The length is the number of sequences that can be formed from the data
    def __len__(self):
        return len(self.data) - self.seq_length

    # Get the input and output sequences
    # The input sequence is of length seq_length and the output is the next value in the series
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        return x, y
    
# Define the LSTM and GRU models
# The LSTM model consists of an LSTM layer followed by a fully connected layer
class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    # The forward method defines the forward pass of the model
    # It takes the input tensor x and passes it through the LSTM layer
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
    
# Define the GRU model
# The GRU model is similar to the LSTM model but uses GRU cells instead of LSTM cells
class GRUModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out
    
# Define the LVImputer class
# This class is responsible for training and using the LSTM and GRU models for imputing missing values
class LVImputer:
    def __init__(self, model_type: str = 'LSTM', input_size: int = 1, hidden_size: int = 64, num_layers: int = 2,
                 seq_length: int = 10, batch_size: int = 32, learning_rate: float = 0.001, num_epochs: int = 100):
        self.model_type = model_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.model = None
        self.scaler = MinMaxScaler()
        self.criterion = nn.MSELoss()
        self.optimizer = None
        if model_type == 'LSTM':
            self.model = LSTMModel(input_size, hidden_size, num_layers)
        elif model_type == 'GRU':
            self.model = GRUModel(input_size, hidden_size, num_layers)
        else:
            raise ValueError("model_type must be either 'LSTM' or 'GRU'")
        
    # Fit the model to the data
    # The fit method takes the data as input and trains the model
    def fit(self, data: np.ndarray):
        # Scale the data
        data = self.scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)
        # Create dataset and dataloader
        dataset = TimeSeriesDataset(data, self.seq_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        # Define optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Training loop
        for epoch in range(self.num_epochs):
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.float().to(device)
                y_batch = y_batch.float().to(device)
                self.optimizer.zero_grad()
                output = self.model(x_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}')

    # Predict missing values using the trained model
    # The predict method takes the data as input and returns the predicted values
    def predict(self, data: np.ndarray) -> np.ndarray:
        # Scale the data
        data = self.scaler.transform(data.reshape(-1, 1)).reshape(-1)
        # Create dataset and dataloader
        dataset = TimeSeriesDataset(data, self.seq_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        # Prediction loop
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for x_batch, _ in dataloader:
                x_batch = x_batch.float().to(device)
                output = self.model(x_batch)
                predictions.append(output.cpu().numpy())
        predictions = np.concatenate(predictions, axis=0)
        # Inverse scale the predictions
        predictions = self.scaler.inverse_transform(predictions)
        return predictions
    
    # Impute missing values in the data
    # The impute method takes the data as input and returns the imputed data
    def impute(self, data: np.ndarray) -> np.ndarray:
        # Identify missing values
        missing_mask = np.isnan(data)
        # Impute missing values
        data[missing_mask] = self.predict(data[~missing_mask])
        return data
    
    # Evaluate the model using RMSE
    # The evaluate method takes the true data and predicted data as input and returns the RMSE
    def evaluate(self, true_data: np.ndarray, predicted_data: np.ndarray) -> float:
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(true_data, predicted_data))
        return rmse
    
    # Save and load the model
    # The save_model method saves the model to the specified path
    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')

    # The load_model method loads the model from the specified path
    # It also sets the model to evaluation mode
    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print(f'Model loaded from {path}')

    # Set and get hyperparameters
    # The set_hyperparameters method allows the user to set hyperparameters for the model
    def set_hyperparameters(self, input_size: int = None, hidden_size: int = None, num_layers: int = None,
                            seq_length: int = None, batch_size: int = None, learning_rate: float = None,
                            num_epochs: int = None):
        if input_size is not None:
            self.input_size = input_size
        if hidden_size is not None:
            self.hidden_size = hidden_size
        if num_layers is not None:
            self.num_layers = num_layers
        if seq_length is not None:
            self.seq_length = seq_length
        if batch_size is not None:
            self.batch_size = batch_size
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if num_epochs is not None:
            self.num_epochs = num_epochs
    
    # The get_hyperparameters method returns the current hyperparameters of the model
    # This is useful in case I am debugging or logging
    def get_hyperparameters(self) -> dict:
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'seq_length': self.seq_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs
        }
    
    # Set and get device
    # The set_device method allows the user to set the device for training and inference
    def set_device(self, device: str):
        if device not in ['cpu', 'cuda']:
            raise ValueError("device must be either 'cpu' or 'cuda'")
        self.device = torch.device(device)
        self.model.to(self.device)
        print(f'Model moved to {self.device}')

    # The get_device method returns the current device of the model
    # This is useful in case I am debugging or logging
    def get_device(self) -> str:
        return 'cuda' if torch.cuda.is_available() else 'cpu'   
    
    # Set and get random seed
    # The set_random_seed method allows the user to set a random seed for reproducibility
    def set_random_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f'Random seed set to {seed}')

    # The get_random_seed method returns the current random seed
    # This is useful in case I am debugging or logging
    def get_random_seed(self) -> int:
        return torch.initial_seed() % (2**32)
    
    # Set and get logging
    # The set_logging method allows the user to set a log file for logging
    def set_logging(self, log_file: str):
        import logging
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s:%(levelname)s:%(message)s')
        print(f'Logging set to {log_file}')

    # The get_logging method returns the current log file
    # This is useful in case I am debugging or logging
    def log(self, message: str):
        import logging
        logging.info(message)
        print(message)

    # Set and get early stopping
    # The set_early_stopping method allows the user to set early stopping criteria
    def set_early_stopping(self, patience: int = 10):
        self.patience = patience
        self.best_loss = float('inf')
        self.early_stopping_counter = 0
        print(f'Early stopping set with patience {patience}')

    # The check_early_stopping method checks if early stopping criteria are met
    # If the loss has not improved for patience epochs, it returns True
    def check_early_stopping(self, loss: float):
        if loss < self.best_loss:
            self.best_loss = loss
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        if self.early_stopping_counter >= self.patience:
            print('Early stopping triggered')
            return True
        return False
    
    # Set and get callbacks
    # The set_callbacks method allows the user to set callbacks for training and inference
    def set_callbacks(self, callbacks: List[callable]):
        self.callbacks = callbacks
        print(f'Callbacks set: {callbacks}')

    # The run_callbacks method runs the callbacks with the given arguments
    # This is useful in case I am debugging or logging 
    def run_callbacks(self, *args, **kwargs):
        for callback in self.callbacks:
            callback(*args, **kwargs)
        print('Callbacks executed')


    def clear_callbacks(self):
        self.callbacks = []
        print('Callbacks cleared')

    def set_logging_level(self, level: str):
        import logging
        levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        if level not in levels:
            raise ValueError("level must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")
        logging.getLogger().setLevel(levels[level])
        print(f'Logging level set to {level}')
        
    def get_logging_level(self) -> str:
        import logging
        levels = {
            logging.DEBUG: 'DEBUG',
            logging.INFO: 'INFO',
            logging.WARNING: 'WARNING',
            logging.ERROR: 'ERROR',
            logging.CRITICAL: 'CRITICAL'
        }
        return levels.get(logging.getLogger().level, 'UNKNOWN')
    
    def set_model_name(self, name: str):
        self.model_name = name
        print(f'Model name set to {name}')

    def get_model_name(self) -> str:
        return self.model_name if hasattr(self, 'model_name') else 'Model name not set'
   
    def set_model_parameters(self, parameters: dict):
        self.model_parameters = parameters
        print(f'Model parameters set to {parameters}')

    def get_model_parameters(self) -> dict:
        return self.model_parameters if hasattr(self, 'model_parameters') else {}