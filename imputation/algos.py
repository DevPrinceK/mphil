import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LSTMImputer(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMImputer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

class DeepImputer:
    def __init__(self, model='LSTM', seed=5, missing_value=np.nan, epochs=100, lr=0.001):
        self.model_type = model
        self.seed = seed
        self.missing_value = missing_value
        self.epochs = epochs
        self.lr = lr
        self.models = {}
    
    def _get_model(self, input_size):
        if self.model_type == 'LSTM':
            return LSTMImputer(input_size)
        else:
            raise ValueError(f"Model {self.model_type} is not supported.")
    
    def _train_model(self, X_train, y_train):
        input_size = X_train.shape[2]
        model = self._get_model(input_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        
        dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        return model
    
    def _impute_column(self, df, column):
        missing_indices = df[column].isnull()
        if missing_indices.sum() == 0:
            return df
        
        for idx in missing_indices[missing_indices].index:
            start_idx = max(0, idx - self.seed)
            X_train = df.loc[start_idx:idx-1].drop(columns=column).dropna().values
            y_train = df.loc[start_idx:idx-1, column].dropna().values

            if len(y_train) == 0:
                continue

            # Reshape for LSTM (batch_size, sequence_length, input_size)
            X_train = X_train.reshape(-1, 1, X_train.shape[1])
            
            model = self._train_model(X_train, y_train)
            X_test = df.loc[idx, df.columns != column].values.reshape(1, 1, -1)
            df.loc[idx, column] = model(torch.tensor(X_test, dtype=torch.float32)).item()
        
        return df
    
    def fit_transform(self, df):
        df = df.copy()
        for column in df.columns:
            df = self._impute_column(df, column)
        return df

# Example usage:
# df = pd.DataFrame(... your data ...)
# imputer = DeepImputer(model='LSTM', seed=5, missing_value=np.nan, epochs=100, lr=0.001)
# imputed_df = imputer.fit_transform(df)