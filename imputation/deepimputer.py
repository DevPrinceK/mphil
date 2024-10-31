import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class DeepImputer:
    def __init__(self, model='MLP', seed=5, missing_value=np.nan, epochs=100):
        self.model_type = model
        self.seed = seed
        self.missing_value = missing_value
        self.epochs = epochs
        self.models = {}
    
    def _get_model(self):
        if self.model_type == 'MLP':
            return make_pipeline(StandardScaler(), MLPRegressor(max_iter=self.epochs))
        else:
            raise ValueError(f"Model {self.model_type} is not supported.")
    
    def _memoize_train(self, X_train, y_train, column):
        if column not in self.models:
            model = self._get_model()
            model.fit(X_train, y_train)
            self.models[column] = model
        return self.models[column]
    
    def _impute_column(self, df, column):
        missing_indices = df[column].isnull()
        if missing_indices.sum() == 0:
            return df
        
        for idx in missing_indices[missing_indices].index:
            start_idx = max(0, idx - self.seed)
            X_train = df.loc[start_idx:idx-1].drop(columns=column).dropna()
            y_train = df.loc[start_idx:idx-1, column].dropna()

            if len(y_train) == 0:
                continue
            
            model = self._memoize_train(X_train, y_train, column)
            X_test = df.loc[idx, X_train.columns].values.reshape(1, -1)
            df.loc[idx, column] = model.predict(X_test)[0]
        
        return df
    
    def fit_transform(self, df):
        df = df.copy()
        for column in df.columns:
            df = self._impute_column(df, column)
        return df

# Example usage:
# df = pd.DataFrame(... your data ...)
# imputer = DeepImputer(model='MLP', seed=5, missing_value=np.nan, epochs=100)
# imputed_df = imputer.fit_transform(df)