import pandas as pd
import json


class DataLoader:
    def __init__(self, path, sep=","):
        self.columns = None
        self.data = None
        self.path = path
        self.sep = sep

    def get_data(self):
        if self.path.endswith('.csv'):
            self.data = pd.read_csv(self.path, sep=self.sep)
        elif self.path.endswith('.json') or self.path.endswith('.jsonl'):
            with open(self.path, 'r') as fp:
                json_lines = [json.loads(line) for line in fp]
            self.data = pd.DataFrame(json_lines)
        self.columns = self.data.columns
        return self.data
