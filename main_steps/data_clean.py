from abc import abstractmethod
from algorithms.dataCleaning import handle_duplicates, handle_missing_values, handle_outliers


class DataClean:
    def __init__(self, data = None):
        self.data = data

    def setData(self, data):
        self.data = data

    @abstractmethod
    def do_cleaning(self):
        pass


class DuplicatesCleaner(DataClean):

    def do_cleaning(self):
        return handle_duplicates(self.data)


class MissingValueCleaner(DataClean):

    def __init__(self, data=None,
                 numerical_columns=[],
                 categorical_columns=[],
                 datetime_columns=[],
                 short_text_columns=[],
                 long_text_columns=[]):
        super().__init__(data)
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.datetime_columns = datetime_columns
        self.short_text_columns = short_text_columns
        self.long_text_columns = long_text_columns

    def do_cleaning(self):
        return handle_missing_values(
            self.data,
            self.numerical_columns,
            self.categorical_columns,
            self.datetime_columns,
            self.short_text_columns,
            self.long_text_columns)


class OutliersCleaner(DataClean):

    def __init__(self, data = None, numerical_columns=[]):
        super().__init__(data)
        self.numerical_columns = numerical_columns

    def do_cleaning(self):
        return handle_outliers(
            self.data,
            self.numerical_columns)