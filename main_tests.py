import pandas as pd

from main_steps.data_clean import DuplicatesCleaner, MissingValueCleaner, OutliersCleaner
from main_steps.data_loader import DataLoader
from main_steps.data_prepare import DataPrepare
from main_steps.data_corruption import DuplicateCorruption, IntroNanCorruption, IntroOutliersCorruption, CorruptionData
from main_steps.model_prepare import RandomForestModelPrepare, NewModelPrepare
from main_steps.model_evaluator import AccuracyEvaluator
import matplotlib.pyplot as plt
import numpy as np


# from main_steps.model_prepare import ModelPrepare

def train_and_evaluate(total_data, train_data, test_data, train_labels, test_labels, target):
    # get a model pipline from the training data
    model_preparation = NewModelPrepare(
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        total_data=total_data,
        categorical_columns=data_prepare.categorical_columns,
        numerical_columns=data_prepare.numerical_columns,
        datetime_columns=data_prepare.datetime_columns,
        text_columns=data_prepare.text_columns,
    )
    #
    result = model_preparation.new_fit(target)
    # # #
    # evaluation = AccuracyEvaluator(model)
    # result = evaluation.evaluate(test_data, test_labels)
    # print(f"- Accuracy: {result:.4f}\n")
    return result


# do test for multiple times and return a list for accuracy.
# The test will be done on original data.
# The model is for predicting the target column
def evaluate_before_corruption(sample_size, test_time, origin_data, target_column):
    result_list = []
    for _ in range(test_time):
        data = origin_data.sample(n=sample_size).copy()
        data_prepare = DataPrepare(data, target_columns=[target_column])
        # data_prepare.classify_column_types()
        data_prepare.set_target(target_columns=[target_column])

        train_data, test_data = data_prepare.split_train_and_test(0.1, 1234)
        train_feature, train_label = data_prepare.split_feature_and_label(train_data)
        test_feature, test_label = data_prepare.split_feature_and_label(test_data)
        single_result = train_and_evaluate(data, train_feature, test_feature, train_label, test_label, target_column)
        result_list.append(single_result)
    return result_list


# do test for multiple times and return a list for accuracy.
# The test will be done on corrupted data.
# The model is for predicting the target column
def evaluate_after_corruption(sample_size, test_time, origin_data, target_column, corruptions):
    result_list = []
    for _ in range(test_time):
        data = origin_data.sample(n=sample_size).copy()
        data_prepare = DataPrepare(data, target_columns=[target_column])
        # data_prepare.classify_column_types()
        data_prepare.set_target(target_columns=[target_column])

        train_data, test_data = data_prepare.split_train_and_test(0.1, 1234)
        corrupted_data = train_data
        for item in corruptions:
            item.setData(corrupted_data)
            corrupted_data = item.do_corrupt()
        train_feature, train_label = data_prepare.split_feature_and_label(corrupted_data)
        test_feature, test_label = data_prepare.split_feature_and_label(test_data)
        # data_evaluator.setData(corrupted_data)
        # data_evaluator.doEvaluate()
        single_result = train_and_evaluate(data, train_feature, test_feature, train_label, test_label, target_column)
        result_list.append(single_result)
    return result_list


# do test for multiple times and return a list for accuracy.
# The test will be done on cleaned data.
# The model is for predicting the target column
# NOTION: if corruptions are empty, then will do cleaners on original data.
def evaluate_after_clean(sample_size, test_time, origin_data, target_column, corruptions, cleaners):
    result_list = []
    for _ in range(test_time):
        data = origin_data.sample(n=sample_size).copy()
        data_prepare = DataPrepare(data, target_columns=[target_column])
        # data_prepare.classify_column_types()
        data_prepare.set_target(target_columns=[target_column])

        train_data, test_data = data_prepare.split_train_and_test(0.1, 1234)
        corrupted_data = train_data
        for item in corruptions:
            item.setData(corrupted_data)
            corrupted_data = item.do_corrupt()
        cleaned_data = corrupted_data
        for item in cleaners:
            item.setData(cleaned_data)
            cleaned_data = item.do_cleaning()

        train_feature, train_label = data_prepare.split_feature_and_label(cleaned_data)
        test_feature, test_label = data_prepare.split_feature_and_label(test_data)
        # data_evaluator.setData(cleaned_data)
        # data_evaluator.doEvaluate()
        single_result = train_and_evaluate(data, train_feature, test_feature, train_label, test_label, target_column)
        result_list.append(single_result)
    return result_list


def single_test(sample_size, origin_data, target_column):
    print(origin_data.shape)
    data = origin_data.sample(n=sample_size).copy()

    columns_str = ', '.join(data.columns)
    print(f"The DataFrame contains the following columns:\n{columns_str}.")
    print("Original data:\n", data.tail(5), "\n")

    data_prepare = DataPrepare(data, target_columns=[target_column])
    # data_prepare.classify_column_types()
    data_prepare.set_target(target_columns=[target_column])

    train_data, test_data = data_prepare.split_train_and_test(0.1, 1234)
    train_feature, train_label = data_prepare.split_feature_and_label(train_data)
    test_feature, test_label = data_prepare.split_feature_and_label(test_data)

    print("====================Before corruption=====================")
    # data_evaluator = TFDVEValuator()
    # data_evaluator.doEvaluate(train_data)
    result_on_origin = train_and_evaluate(data, train_feature, test_feature, train_label, test_label, target_column)

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
    # data_evaluator.doEvaluate(corrupted_data)
    result_on_corrupted = train_and_evaluate(data, train_feature, test_feature, train_label, test_label, target_column)

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
    # data_evaluator.doEvaluate(cleaned_data)
    result_on_cleaned = train_and_evaluate(data, train_feature, test_feature, train_label, test_label, target_column)
    return result_on_origin, result_on_corrupted, result_on_cleaned


if __name__ == '__main__':
    # load data from a path
    loader = DataLoader('data/Amazon Product Reviews/Video_Games_5.json')
    # loader = DataLoader('data/Amazon Product Reviews/video_games.csv')
    data = loader.get_data()
    # analysis the data structure
    target_column = 'verified'
    data_prepare = DataPrepare(data.sample(n=500), target_columns=[target_column])

    # all corruptions
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

    # all cleaners
    duplicate_cleaner = DuplicatesCleaner()
    missing_cleaner = MissingValueCleaner(
        numerical_columns=data_prepare.numerical_columns,
        categorical_columns=data_prepare.categorical_columns,
        short_text_columns=data_prepare.text_columns)
    outlier_cleaner = OutliersCleaner(
        numerical_columns=data_prepare.numerical_columns)

    final_results = []
    result_columns = []
    corruptions = []
    cleaners = []

    # get accuracy list on original data by doing test for 20 times.
    before_clean = evaluate_before_corruption(4000, 20, data, target_column)
    print(before_clean)

    final_results.append(before_clean)
    result_columns.append('origin')
    print(f'origin results: {before_clean}')

    # draw a boxplot with the result
    plt.boxplot(np.array(before_clean))
    plt.title(f'Target: {target_column}. Original Data')
    plt.ylabel('accuracy')
    plt.show()

    corruptions.append(duplicat)
    corruptions.append(introNan)
    corruptions.append(introOutlier)

    # get accuracy list on corrupted data by doing test for 20 times.
    after_corruption = evaluate_after_corruption(4000,20, data, target_column, corruptions)
    print(after_corruption)

    final_results.append(after_corruption)
    result_columns.append('corrupted')
    print(f'corrupted results: {after_corruption}')

    plt.boxplot(np.array(after_corruption))
    plt.title(f'Target: {target_column}. After All Corruptions')
    plt.ylabel('accuracy')
    plt.show()

    cleaners.append(duplicate_cleaner)
    cleaners.append(missing_cleaner)
    cleaners.append(outlier_cleaner)
    # get accuracy list on corrupted data by doing test for 20 times.
    after_clean = evaluate_after_clean(4000,20, data, target_column, corruptions, cleaners)
    print(after_clean)

    final_results.append(after_clean)
    result_columns.append('cleaned')
    print(f'cleaned results: {after_clean}')

    plt.boxplot(np.array(after_clean))
    plt.title(f'Target: {target_column}. After All Cleaners')
    plt.ylabel('accuracy')
    plt.show()

    # if you want to test on single corruption:
    corruptions.clear()
    corruptions.append(duplicat)
    single_corruption = evaluate_after_corruption(4000, 20, data, target_column, corruptions)

    final_results.append(single_corruption)
    result_columns.append('duplicat')
    print(f'duplicat results: {single_corruption}')

    plt.boxplot(np.array(single_corruption))
    plt.title(f'Target: {target_column}. After Duplicate Corruption')
    plt.ylabel('accuracy')
    plt.show()

    corruptions.clear()
    corruptions.append(introNan)
    single_corruption = evaluate_after_corruption(4000, 20, data, target_column, corruptions)

    final_results.append(single_corruption)
    result_columns.append('introNan')
    print(f'introNan results: {single_corruption}')

    plt.boxplot(np.array(single_corruption))
    plt.title(f'Target: {target_column}. After Missing Corruption')
    plt.ylabel('accuracy')
    plt.show()

    corruptions.clear()
    corruptions.append(introOutlier)
    single_corruption = evaluate_after_corruption(4000, 20, data, target_column, corruptions)

    final_results.append(single_corruption)
    result_columns.append('introOutlier')
    print(f'introOutlier results: {single_corruption}')

    plt.boxplot(np.array(single_corruption))
    plt.title(f'Target: {target_column}. After Intro Outliers Corruption')
    plt.ylabel('accuracy')
    plt.show()

    # if you want to test on single cleaner:
    corruptions.clear()
    corruptions.append(duplicat)
    cleaners.clear()
    cleaners.append(duplicate_cleaner)
    single_clean = evaluate_after_clean(4000, 20, data, target_column, corruptions, cleaners)

    final_results.append(single_clean)
    result_columns.append('duplicate_cleaner')
    print(f'duplicate_cleaner results: {single_clean}')

    plt.boxplot(np.array(single_clean))
    plt.title(f'Target: {target_column}. After Duplicate Cleaned')
    plt.ylabel('accuracy')
    plt.show()

    corruptions.clear()
    corruptions.append(introNan)
    cleaners.clear()
    cleaners.append(missing_cleaner)
    single_clean = evaluate_after_clean(4000, 20, data, target_column, corruptions, cleaners)

    final_results.append(single_clean)
    result_columns.append('missing_cleaner')
    print(f'missing_cleaner results: {single_clean}')

    plt.boxplot(np.array(single_clean))
    plt.title(f'Target: {target_column}. After Missing Values Cleaned')
    plt.ylabel('accuracy')
    plt.show()

    corruptions.clear()
    corruptions.append(introOutlier)
    cleaners.clear()
    cleaners.append(outlier_cleaner)
    single_clean = evaluate_after_clean(4000, 20, data, target_column, corruptions, cleaners)

    final_results.append(single_clean)
    result_columns.append('outlier_cleaner')
    print(f'outlier_cleaner results: {single_clean}')

    plt.boxplot(np.array(single_clean))
    plt.title(f'Target: {target_column}. After Outliers Cleaned')
    plt.ylabel('accuracy')
    plt.show()

    # if you want to test cleaners on original data
    corruptions.clear()
    cleaners.clear()
    cleaners.append(duplicate_cleaner)
    single_clean = evaluate_after_clean(4000, 20, data, target_column, corruptions, cleaners)

    final_results.append(single_clean)
    result_columns.append('direct_duplicate_cleaner')
    print(f'direct_duplicate_cleaner results: {single_clean}')

    plt.boxplot(np.array(single_clean))
    plt.title(f'Target: {target_column}. Directly Duplicate Cleaned')
    plt.ylabel('accuracy')
    plt.show()

    corruptions.clear()
    cleaners.clear()
    cleaners.append(missing_cleaner)
    single_clean = evaluate_after_clean(4000, 20, data, target_column, corruptions, cleaners)

    final_results.append(single_clean)
    result_columns.append('direct_missing_cleaner')
    print(f'direct_missing_cleaner results: {single_clean}')

    plt.boxplot(np.array(single_clean))
    plt.title(f'Target: {target_column}. Directly Values Cleaned')
    plt.ylabel('accuracy')
    plt.show()

    corruptions.clear()
    cleaners.clear()
    cleaners.append(outlier_cleaner)
    single_clean = evaluate_after_clean(4000, 20, data, target_column, corruptions, cleaners)

    final_results.append(single_clean)
    result_columns.append('direct_outlier_cleaner')
    print(f'direct_outlier_cleaner results: {single_clean}')

    plt.boxplot(np.array(single_clean))
    plt.title(f'Target: {target_column}. Directly Outliers Cleaned')
    plt.ylabel('accuracy')
    plt.show()

    plt.boxplot(np.array(single_corruption))
    plt.title(f'Target: {target_column}. After Missing Corruption')
    plt.ylabel('accuracy')
    plt.show()

    corruptions.clear()
    corruptions.append(introOutlier)
    single_corruption = evaluate_after_corruption(4000, 20, data, target_column, corruptions)

    final_results.append(single_corruption)
    result_columns.append('introOutlier')

    plt.boxplot(np.array(single_corruption))
    plt.title(f'Target: {target_column}. After Intro Outliers Corruption')
    plt.ylabel('accuracy')
    plt.show()

    # if you want to test on single cleaner:
    corruptions.clear()
    corruptions.append(duplicat)
    cleaners.clear()
    cleaners.append(duplicate_cleaner)
    single_clean = evaluate_after_clean(4000, 20, data, target_column, corruptions, cleaners)

    final_results.append(single_clean)
    result_columns.append('duplicate_cleaner')

    plt.boxplot(np.array(single_clean))
    plt.title(f'Target: {target_column}. After Duplicate Cleaned')
    plt.ylabel('accuracy')
    plt.show()

    corruptions.clear()
    corruptions.append(introNan)
    cleaners.clear()
    cleaners.append(missing_cleaner)
    single_clean = evaluate_after_clean(4000, 20, data, target_column, corruptions, cleaners)

    final_results.append(single_clean)
    result_columns.append('missing_cleaner')

    plt.boxplot(np.array(single_clean))
    plt.title(f'Target: {target_column}. After Missing Values Cleaned')
    plt.ylabel('accuracy')
    plt.show()

    corruptions.clear()
    corruptions.append(introOutlier)
    cleaners.clear()
    cleaners.append(outlier_cleaner)
    single_clean = evaluate_after_clean(4000, 20, data, target_column, corruptions, cleaners)

    final_results.append(single_clean)
    result_columns.append('outlier_cleaner')

    plt.boxplot(np.array(single_clean))
    plt.title(f'Target: {target_column}. After Outliers Cleaned')
    plt.ylabel('accuracy')
    plt.show()

    # if you want to test cleaners on original data
    corruptions.clear()
    cleaners.clear()
    cleaners.append(duplicate_cleaner)
    single_clean = evaluate_after_clean(4000, 20, data, target_column, corruptions, cleaners)

    final_results.append(single_clean)
    result_columns.append('direct_duplicate_cleaner')

    plt.boxplot(np.array(single_clean))
    plt.title(f'Target: {target_column}. Directly Duplicate Cleaned')
    plt.ylabel('accuracy')
    plt.show()

    corruptions.clear()
    cleaners.clear()
    cleaners.append(missing_cleaner)
    single_clean = evaluate_after_clean(4000, 20, data, target_column, corruptions, cleaners)

    final_results.append(single_clean)
    result_columns.append('direct_missing_cleaner')

    plt.boxplot(np.array(single_clean))
    plt.title(f'Target: {target_column}. Directly Values Cleaned')
    plt.ylabel('accuracy')
    plt.show()

    corruptions.clear()
    cleaners.clear()
    cleaners.append(outlier_cleaner)
    single_clean = evaluate_after_clean(4000, 20, data, target_column, corruptions, cleaners)

    final_results.append(single_clean)
    result_columns.append('direct_outlier_cleaner')

    plt.boxplot(np.array(single_clean))
    plt.title(f'Target: {target_column}. Directly Outliers Cleaned')
    plt.ylabel('accuracy')
    plt.show()

    fractions = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    for i in fractions:
        introNan = IntroNanCorruption(howto=CorruptionData(
            column=[data_prepare.numerical_columns,
                    data_prepare.categorical_columns,
                    data_prepare.datetime_columns,
                    data_prepare.text_columns],
            fraction=i
        ))
        corruptions.clear()
        corruptions.append(introNan)
        cleaners.clear()
        cleaners.append(missing_cleaner)
        single_clean = evaluate_after_clean(4000, 20, data, target_column, corruptions, cleaners)
        final_results.append(single_clean)
        result_columns.append(f'missing_cleaner_{i}')
        print(f'missing_cleaner_{i} results: {single_clean}')

    results_df = pd.DataFrame(final_results).T
    results_df.columns = result_columns
    print(final_results)
    # do a full test for one time.
    # (single_origin, single_corrupted, single_cleaned) = single_test(4000, data, target_column)
