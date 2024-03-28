import pandas as pd
from pyspark.sql import SparkSession
import pydeequ
from collections import defaultdict
import os


class DataQualityEvaluator:
    def __init__(self, spark_session):
        self.spark_session = spark_session

    def validate_data(self, data):
        checks = [
            self.check_completeness(data),
            self.check_accuracy(data),
            self.check_consistency(data),
        ]

        results = self.run_checks(data, checks)
        return results

    def run_checks(self, data, checks):
        results = []

        for check in checks:
            result = check(data)
            results.append(result)

        return results

    def check_completeness(self, data):
        def check(data):
            result = defaultdict(dict)

            # Check missing value proportion in each column
            missing_proportion = data.isnull().mean()
            result['missing_proportion'] = missing_proportion.to_dict()

            # Check unique value count in each column
            unique_count = data.nunique()
            result['unique_count'] = unique_count.to_dict()

            # Check total missing values count
            total_missing = data.isnull().sum().sum()
            result['total_missing'] = total_missing

            return result

        return check

    def check_accuracy(self, data):
        def check(data):
            result = defaultdict(dict)

            # Check numerical columns: min, max, mean
            numerical_stats = data.describe().loc[['min', 'max', 'mean']]
            for col in data.select_dtypes(include='number').columns:
                result['numerical_stats'][col] = {
                    'min': numerical_stats.loc['min', col],
                    'max': numerical_stats.loc['max', col],
                    'mean': numerical_stats.loc['mean', col]
                }

            # Check categorical columns: value counts
            categorical_cols = data.select_dtypes(exclude='number').columns
            value_counts = {col: data[col].value_counts() for col in categorical_cols}
            for col in categorical_cols:
                result['categorical_value_counts'][col] = value_counts[col].to_dict()

        return check

    def check_consistency(self, data):
        def check(data):
            result = defaultdict(dict)

            # Check for duplicate rows
            duplicate_rows = data.duplicated().sum()
            result['duplicate_rows'] = duplicate_rows

            # Check for outliers
            outlier_threshold = 1.5
            numerical_cols = data.select_dtypes(include='number').columns
            for col in numerical_cols:
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - outlier_threshold * iqr
                upper_bound = q3 + outlier_threshold * iqr
                num_outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                result['outliers'][col] = num_outliers

            return result

        return check


if __name__ == "__main__":
    # Load data
    data = pd.read_csv('video_games.csv')

    # Create a SparkSession
    spark = SparkSession.builder \
        .appName("DataValidation") \
        .config("spark.jars.packages", pydeequ.deequ_maven_coord) \
        .config("spark.jars.excludes", pydeequ.f2j_maven_coord) \
        .getOrCreate()

    # Create DataQualityEvaluator
    evaluator = DataQualityEvaluator(spark)

    # Validate data
    validation_results = evaluator.validate_data(data)

    # Print the results
    for result in validation_results:
        print(result)

    # Stop SparkSession
    spark.stop()
