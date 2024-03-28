from typing import List
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class DataPrepare:
    def __init__(self, data, target_columns = []):
        self.data = data
        self.target_columns = target_columns
        self.text_columns = []
        self.categorical_columns = []
        self.numerical_columns = []
        self.datetime_columns = []

        self.classify_column_types() # Automatically classify column types upon initialization

    def set_columns(
        self,
        categorical_columns: List[str] = [],
        numerical_columns: List[str] = [],
        text_columns: List[str] = [],
        target_columns: List[str] = [],
        datetime_columns: List[str] = []  # 添加 datetime_columns 参数
    ):
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.text_columns = text_columns
        self.target_columns = target_columns
        self.datetime_columns = datetime_columns

    def set_target(self, target_columns):
        self.target_columns = target_columns

    def classify_column_types(self):
        unique_threshold = 0.2  # Threshold for unique values to consider a column categorical
        word_count_threshold = 30  # Threshold for word count to distinguish between short and long text

        for col in self.data.columns:
            if col in self.target_columns:
                continue
            sample_size = min(1000, len(self.data[col].dropna()))
            sample_data = self.data[col].dropna().sample(n=sample_size, random_state=42)

            converted_sample = pd.to_datetime(sample_data, errors='coerce', format='%m %d, %Y')
            # If all sampled non-empty values can be parsed as datetime, consider the column as datetime
            if not converted_sample.isna().any():
                if sample_data.dtype in ['int64', 'float64']:
                    # For numeric columns, check if values are in the typical Unix timestamp range
                    if sample_data.between(1e9, 2e9).any():
                        self.datetime_columns.append(col)
                        continue  # Skip further checks for this column
                else:
                    # Non-numeric columns parsed as datetime are added to datetime columns
                    self.datetime_columns.append(col)
                    continue

            # For non-datetime columns, determine other types
            if self.data[col].dtype in ['int64', 'float64']:
                # Determine if a numeric column is categorical based on unique values and threshold
                proportion_unique = self.data[col].nunique() / self.data[col].notnull().sum()
                if self.data[col].nunique() <= 4 and proportion_unique <= unique_threshold:
                    self.categorical_columns.append(col)
                else:
                    self.numerical_columns.append(col)

            elif self.data[col].dtype in ['bool', 'category']:
                self.categorical_columns.append(col)

            elif self.data[col].dtype in ['object', 'string']:
                self.data[col] = self.data[col].astype(str).replace('nan', np.nan) # Ensure all data is string for analysis
                sample_data = sample_data.astype(str) # Ensure sample data is string for word count
                proportion_unique = self.data[col].nunique() / self.data[col].notnull().sum()
                if self.data[col].nunique() <= 100 and proportion_unique <= unique_threshold:
                    self.categorical_columns.append(col)
                else:
                    # Calculate word count for text analysis
                    word_counts = sample_data.str.split().str.len().fillna(0)
                    max_word_count = word_counts.max()
                    if max_word_count <= word_count_threshold:
                        self.text_columns.append(col)  # Consider as short text for simplicity
                    else:
                        self.text_columns.append(col)  # Consider as long text for simplicity
        for col in self.categorical_columns:
            self.data[col] = self.data[col].astype('category')
        for col in self.numerical_columns:
            self.data[col] = self.data[col].astype('float64')

    def split_train_and_test(self, test_size, seed):
        np.random.seed(seed)
        data = self.data[self.text_columns + self.numerical_columns + self.categorical_columns +
                         self.target_columns].dropna()
        train_data, test_data = train_test_split(
            data, test_size=test_size, random_state=seed)
        return train_data, test_data

    def split_feature_and_label(self, data):
        new_data = data.copy().dropna()
        target = new_data.pop(self.target_columns[0])
        return new_data, target
