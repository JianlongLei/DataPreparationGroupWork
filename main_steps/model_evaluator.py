from abc import abstractmethod


class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def evaluate(self, train_data, test_data):
        pass


class AccuracyEvaluator(ModelEvaluator):

    def evaluate(self, test_data, test_labels):
        return self.model.score(test_data, test_labels)