import pandas as pd
import random
import numpy as np

class ReplaceCharacter:
    def __init__(self, column, fraction, rate):
        self.column = column  
        self.fraction = fraction  # how many rows will be replaced
        self.rate = rate  # how many characters in a row will be replaced

    def _replace_chars_in_text(self, text):
        if not text:
            return text
        
        num_chars_to_replace = max(1, int(len(text) * self.rate))
        text_list = list(text)
        
        # choose a character randomly
        for _ in range(num_chars_to_replace):
            idx_to_replace = random.randint(0, len(text_list) - 1)
            text_list[idx_to_replace] = chr(random.randint(32, 126))
        
        return ''.join(text_list)

    def transform(self, df):

        if self.column not in df.columns:
            raise ValueError(f"Column '{self.column}' not found in DataFrame")
        
        rows_to_replace = np.random.rand(len(df)) < self.fraction
        df.loc[rows_to_replace, self.column] = df.loc[rows_to_replace, self.column].apply(self._replace_chars_in_text)
        
        return df