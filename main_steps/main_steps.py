# from main_steps.data_clean import DataClean
# from main_steps.data_corruption import DataCorruption
import sys
sys.path.append('/Users/tangzj/Desktop/DataPreparationGroupWork')

from data_loader import DataLoader
from data_prepare import DataPrepare
from data_corruption import MissingCorruption, MissingCorruptionData, ReplaceCharacterCorruption, \
    ReplaceCharactorCorruptionData
# from model_prepare_gpu import GPUClassificationPrepare
from model_prepare import RandomForestModelPrepare
from model_evaluator import ModelEvaluator, AccuracyEvaluator


# from main_steps.model_prepare import ModelPrepare

def train_and_evaluate(data):
    # prepare data, like mark data type, training data columns
    # and target labels
    data_prepare = DataPrepare(data)
    # data_prepare.classify_column_types()
    data_prepare.set_columns(
        categorical_columns=['verified'],
        numerical_columns=[],
        text_columns=['reviewText', 'summary'],
        target_columns=['overall']
    )
    # data_prepare.set_target(target_columns=['overall'])

    train_data, test_data, train_labels, test_labels = data_prepare.get_training_data(0.3, 1234)

    # get a model pipline from the training data
    model_preparation = RandomForestModelPrepare(
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        categorical_columns=data_prepare.categorical_columns,
        numerical_columns=data_prepare.numerical_columns,
        text_columns=data_prepare.text_columns,
    )
    # model_preparation = GPUClassificationPrepare(
    #     train_data=train_data,
    #     train_labels=train_labels,
    #     test_data=test_data,
    #     test_labels=test_labels,
    #     categorical_columns=data_prepare.categorical_columns,
    #     numerical_columns=data_prepare.numerical_columns,
    #     text_columns=data_prepare.text_columns,
    # )

    #
    model = model_preparation.fit()
    # #
    evaluation = AccuracyEvaluator(model)
    result = evaluation.evaluate(test_data, test_labels)
    print(f"- Accuracy: {result:.4f}\n")


if __name__ == '__main__':
    # load data from a path
    loader = DataLoader('../data/Amazon Product Reviews/test_1000.csv')
    # loader = DataLoader('../data/Amazon Product Reviews/test.csv')
    data = loader.get_data()
    
    columns_str = ', '.join(data.columns) 
    print(f"The DataFrame contains the following columns:\n{columns_str}.")
    print("Original data:\n", data.tail(5), "\n")

    print("====================Before corruption=====================")
    train_and_evaluate(data)
    # do corruptions
    # corruptions = []
    missing = MissingCorruption(data, howto=MissingCorruptionData(
        column='verified',
        fraction=0.3,
        missing_method='MCAR'
    ))
    # corruptions.append(missing)
    #
    # replace = ReplaceCharacterCorruption(data, ReplaceCharactorCorruptionData(
    #     column='reviewText',
    #     fraction=0.2,
    #     rate=0.3
    # ))
    
    print("====================After corruption=====================")
    data = missing.do_corrupt()
    train_and_evaluate(data)

    #
    # data_clean = DataClean()
    # data_clean.do_cleaning()
    # evaluation.evaluate(model_prepare.fit_baseline_model())
