from abc import abstractmethod


class DataClean:
    def __init__(self, data):
        self.data = data

    @abstractmethod
    def do_cleaning(self):
        pass