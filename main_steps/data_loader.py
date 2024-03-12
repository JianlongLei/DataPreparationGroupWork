import pandas as pd


class DataLoader:
    def __init__(self, path, sep=","):
        self.columns = None
        self.data = None
        self.path = path
        self.sep = sep

    def get_data(self):
        self.data = pd.read_csv(self.path, sep=self.sep)
        self.columns = self.data.columns
        return self.data
