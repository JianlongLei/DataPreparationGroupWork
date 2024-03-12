from abc import abstractmethod
from random import random

import numpy as np

from corruptions.missing_value import MissingValue
from corruptions.replace_character import ReplaceCharacter


class CorruptionData:
    def __init__(self, column, fraction):
        self.column = column
        self.fraction = fraction


class DataCorruption:
    def __init__(self, data, howto):
        self.data = data
        self.howto = howto

    def get_dtype(self, df):
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
        return numeric_cols, non_numeric_cols

    def sample_rows(self, data):

        if self.fraction == 1.0:
            rows = data.index
        # Completely At Random
        elif self.sampling.endswith('CAR'):
            rows = np.random.permutation(data.index)[:int(len(data) * self.fraction)]
        elif self.sampling.endswith('NAR') or self.sampling.endswith('AR'):
            n_values_to_discard = int(len(data) * min(self.fraction, 1.0))
            perc_lower_start = np.random.randint(0, len(data) - n_values_to_discard)
            perc_idx = range(perc_lower_start, perc_lower_start + n_values_to_discard)

            # Not At Random
            if self.sampling.endswith('NAR'):
                # pick a random percentile of values in this column
                rows = data[self.column].sort_values().iloc[perc_idx].index

            # At Random
            elif self.sampling.endswith('AR'):
                depends_on_col = np.random.choice(list(set(data.columns) - {self.column}))
                # pick a random percentile of values in other column
                rows = data[depends_on_col].sort_values().iloc[perc_idx].index

        else:
            ValueError(f"sampling type '{self.sampling}' not recognized")

        return rows

    @abstractmethod
    def do_corrupt(self):
        pass


class MissingCorruptionData(CorruptionData):

    def __init__(self, column, fraction, missing_method):
        super().__init__(column, fraction)
        self.missing_method = missing_method


class MissingCorruption(DataCorruption):

    def __init__(self, data, howto: MissingCorruptionData):
        super().__init__(data, howto)

    def do_corrupt(self):
        column = self.howto.column
        fraction = self.howto.fraction
        method = self.howto.missing_method
        pipeline = MissingValue(column=column, fraction=fraction, missing_method=method, na_value=np.nan)
        pipeline.apply_missing(self.data)
        return self.data



class ReplaceCharactorCorruptionData(CorruptionData):

    def __init__(self, column, fraction, rate):
        super().__init__(column, fraction)
        self.rate = rate

class ReplaceCharacterCorruption(DataCorruption):

    def __init__(self, data, howto : ReplaceCharactorCorruptionData):
        super().__init__(data, howto)

    def do_corrupt(self):
        column = self.howto.column
        fraction = self.howto.fraction
        rate = self.howto.rate
        pipeline = ReplaceCharacter(column=column, fraction=fraction, rate=rate)

        # 应用pipeline
        df_transformed = pipeline.transform(self.data)
        return df_transformed

# class GaussianNoise(DataCorruption):
#
#     def do_corrupt(self):
#         data = self.data
#         column = self.howto.column
#         df = data.copy(deep=True)
#         stddev = np.std(df[columns])
#         scale = random.uniform(1, 5)
#
#         if self.fraction > 0:
#             rows = self.sample_rows(data)
#             noise = np.random.normal(0, scale * stddev, size=len(rows))
#             df.loc[rows, column] += noise
#
#         return df
