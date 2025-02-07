from sklearn.preprocessing import StandardScaler
import pandas as pd
from config import Config

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def handle_missing_values(self, df):
        """Handles missing values using configured strategy."""
        if Config.MISSING_VALUE_STRATEGY == "mean":
            return df.fillna(df.mean())
        elif Config.MISSING_VALUE_STRATEGY == "median":
            return df.fillna(df.median())
        else:
            return df.dropna()

    def scale_features(self, X_train, X_test):
        """Scales features using StandardScaler."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
