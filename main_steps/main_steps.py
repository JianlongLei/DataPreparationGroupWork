# from main_steps.data_clean import DataClean
# from main_steps.data_corruption import DataCorruption
import sys

from data_clean import DuplicatesCleaner, MissingValueCleaner, OutliersCleaner

sys.path.append('/Users/tangzj/Desktop/DataPreparationGroupWork')

from data_loader import DataLoader
from data_prepare import DataPrepare
from data_corruption import MissingCorruption, MissingCorruptionData, ReplaceCharacterCorruption, \
    ReplaceCharactorCorruptionData, DuplicateCorruption, IntroNanCorruption, IntroOutliersCorruption, DataCorruption, \
    CorruptionData
# from model_prepare_gpu import GPUClassificationPrepare
from model_prepare import RandomForestModelPrepare
from model_evaluator import ModelEvaluator, AccuracyEvaluator


# from main_steps.model_prepare import ModelPrepare

def train_and_evaluate(train_data, test_data, train_labels, test_labels):
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
    #
    model = model_preparation.fit()
    # #
    evaluation = AccuracyEvaluator(model)
    result = evaluation.evaluate(test_data, test_labels)
    print(f"- Accuracy: {result:.4f}\n")


if __name__ == '__main__':
    # load data from a path
    loader = DataLoader('../data/Amazon Product Reviews/video_games.csv')
    # loader = DataLoader('../data/Amazon Product Reviews/test_1000.csv')
    data = loader.get_data()

    columns_str = ', '.join(data.columns)
    print(f"The DataFrame contains the following columns:\n{columns_str}.")
    print("Original data:\n", data.tail(5), "\n")

    data_prepare = DataPrepare(data, target_columns=['overall'])
    # data_prepare.classify_column_types()
    data_prepare.set_target(target_columns=['overall'])

    train_data, test_data = data_prepare.split_train_and_test(0.3, 1234)
    train_feature, train_label = data_prepare.split_feature_and_label(train_data)
    test_feature, test_label = data_prepare.split_feature_and_label(test_data)

    print("====================Before corruption=====================")
    # data_evaluator = DeequEvaluator()
    # data_evaluator.setData(train_data)
    # data_evaluator.doEvaluate()
    train_and_evaluate(train_feature, test_feature, train_label, test_label)

    # do corruptions
    corruptions = []
    duplicat = DuplicateCorruption(howto=CorruptionData())
    introNan = IntroNanCorruption(howto=CorruptionData(
        column=[data_prepare.numerical_columns,
                data_prepare.categorical_columns,
                data_prepare.datetime_columns,
                data_prepare.text_columns],
        fraction=0.02
    ))
    introOutlier = IntroOutliersCorruption(howto=CorruptionData(
        column=data_prepare.numerical_columns
    ))
    corruptions.append(duplicat)
    corruptions.append(introNan)
    corruptions.append(introOutlier)

    corrupted_data = train_data
    for item in corruptions:
        item.setData(corrupted_data)
        corrupted_data = item.do_corrupt()

    print("====================After corruption=====================")
    train_feature, train_label = data_prepare.split_feature_and_label(corrupted_data)
    test_feature, test_label = data_prepare.split_feature_and_label(test_data)
    # data_evaluator.setData(corrupted_data)
    # data_evaluator.doEvaluate()
    train_and_evaluate(train_feature, test_feature, train_label, test_label)

    # clean data
    cleaners = []
    duplicate_cleaner = DuplicatesCleaner()
    missing_cleaner = MissingValueCleaner(
        numerical_columns=data_prepare.numerical_columns,
        categorical_columns=data_prepare.categorical_columns,
        short_text_columns=data_prepare.text_columns)
    outlier_cleaner = OutliersCleaner(
        numerical_columns=data_prepare.numerical_columns)
    cleaners.append(duplicate_cleaner)
    cleaners.append(missing_cleaner)
    cleaners.append(outlier_cleaner)

    cleaned_data = corrupted_data
    for item in cleaners:
        item.setData(cleaned_data)
        cleaned_data = item.do_cleaning()

    print("====================After clean=====================")
    train_feature, train_label = data_prepare.split_feature_and_label(cleaned_data)
    test_feature, test_label = data_prepare.split_feature_and_label(test_data)
    # data_evaluator.setData(cleaned_data)
    # data_evaluator.doEvaluate()
    train_and_evaluate(train_feature, test_feature, train_label, test_label)

    # data_evaluator.stopSession()
