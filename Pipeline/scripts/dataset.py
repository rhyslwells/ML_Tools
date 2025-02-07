import pandas as pd
from sklearn.model_selection import train_test_split
from config import Config

class Dataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Loads dataset from CSV."""
        self.data = pd.read_csv(self.file_path)
        return self.data

    def split(self, target_column):
        """Splits data into train and test sets."""
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        return train_test_split(X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE)
