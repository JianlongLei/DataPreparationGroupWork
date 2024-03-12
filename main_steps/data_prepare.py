from typing import List
from sklearn.model_selection import train_test_split
import numpy as np


class DataPrepare:
    def __init__(self, data):
        self.target_columns = []
        self.text_columns = []
        self.categorical_columns = []
        self.numerical_columns = []
        self.data = data

    def set_columns(
            self,
            categorical_columns: List[str] = [],
            numerical_columns: List[str] = [],
            text_columns: List[str] = [],
            target_columns: List[str] = []
    ):
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.text_columns = text_columns
        self.target_columns = target_columns

    def get_training_data(self, test_size, seed):
        np.random.seed(seed)
        data = self.data[self.text_columns + self.numerical_columns + self.categorical_columns + self.target_columns].dropna()
        target = data.pop(self.target_columns[0])
        train_data, test_data, train_labels, test_labels = train_test_split(
            data, target, test_size=test_size, random_state=seed)
        return train_data, test_data, train_labels, test_labels
