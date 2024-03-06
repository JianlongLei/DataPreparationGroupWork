import numpy as np
import pandas as pd
from pyspark.sql.functions import rand, when, row_number, lit
from pyspark.sql.window import Window


class MissingValuePipeline:
    def __init__(self, column, fraction, missing_method, na_value=np.nan):
        self.column = column
        self.fraction = fraction
        self.missing_method = missing_method
        self.na_value = na_value

    def apply_missing(self, df):
        if self.missing_method == 'MCAR':
            self._apply_mcar(df)
        elif self.missing_method == 'MAR':
            self._apply_mar(df)
        elif self.missing_method == 'MNAR':
            self._apply_mnar(df)
        else:
            raise ValueError("Invalid missing method. Choose 'MCAR', 'MAR', or 'MNAR'.")

    def _apply_mcar(self, df):
        # MCAR: Assuming missingness completely at random
        missing_indices = np.random.choice(df.index, size=int(len(df) * self.fraction), replace=False)
        df.loc[missing_indices, self.column] = self.na_value

    def _apply_mar(self, df):
        # MAR: Assuming missingness depends on another column
        # choose the next column as a dependent variable
        dependent_col = df.columns[(df.columns.get_loc(self.column) + 1) % len(df.columns)]
        sorted_indices = df.sort_values(by=dependent_col).index
        missing_indices = sorted_indices[:int(len(df) * self.fraction)]
        df.loc[missing_indices, self.column] = self.na_value

    def _apply_mnar(self, df):
        # MNAR: Assuming missingness depends on the values of the column itself
        # not using entropy, just simply testing 
        threshold = df[self.column].quantile(1 - self.fraction)
        missing_indices = df[df[self.column] > threshold].index
        df.loc[missing_indices, self.column] = self.na_value
