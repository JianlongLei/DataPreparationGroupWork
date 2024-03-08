import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from xgboost import XGBRegressor
# from dataIntegration import select_the_dataset
from dataCleaning import *
# from dataPreprocessing import *
import json

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 500)
pd.set_option('display.width', 5000)

with open("data/Amazon Product Reviews/Video_Games_5.json") as f:
    Video_Games = [json.loads(line) for line in f]
df = pd.DataFrame(Video_Games)
# X = df.copy()
num_rows = df.shape[0] // 100
X = df.iloc[:num_rows].copy()
print("Original data:\n", X.tail(10), "\n")


def classify_column_types(df):
    numerical_columns = []
    categorical_columns = []
    datetime_columns = []
    short_text_columns = []
    long_text_columns = []

    unique_threshold = 0.2  # Threshold for unique values to consider a column categorical
    word_count_threshold = 30  # Threshold for word count to distinguish between short and long text

    # First, identify columns that can be parsed as datetime
    for col in df.columns:
        # Sample non-empty values for efficiency
        sample_size = min(1000, len(df[col].dropna()))
        sample_data = df[col].dropna().sample(n=sample_size)

        converted_sample = pd.to_datetime(sample_data, errors='coerce')
        # If all sampled non-empty values can be parsed as datetime, consider the column as datetime
        if not converted_sample.isna().any():
            if sample_data.dtype in ['int64', 'float64']:
                # For numeric columns, check if values are in the typical Unix timestamp range
                if sample_data.between(1e9, 2e9).any():
                    datetime_columns.append(col)
                    continue  # Skip further checks for this column
            else:
                # Non-numeric columns parsed as datetime are added to datetime columns
                datetime_columns.append(col)
                continue

        # For non-datetime columns, determine other types
        if df[col].dtype in ['int64', 'float64']:
            # Determine if a numeric column is categorical based on unique values and threshold
            proportion_unique = df[col].nunique() / df[col].notnull().sum()
            if df[col].nunique() <= 100 and proportion_unique <= unique_threshold:
                categorical_columns.append(col)
            else:
                numerical_columns.append(col)

        elif df[col].dtype in ['bool', 'category']:
            categorical_columns.append(col)

        elif df[col].dtype in ['object', 'string']:
            df[col] = df[col].astype(str).replace('nan', np.nan)  # Ensure all data is string for analysis
            sample_data = sample_data.astype(str)  # Ensure sample data is string for word count
            proportion_unique = df[col].nunique() / df[col].notnull().sum()
            if df[col].nunique() <= 100 and proportion_unique <= unique_threshold:
                categorical_columns.append(col)
            else:
                # Calculate word count for text analysis
                word_counts = sample_data.str.split().str.len().fillna(0)
                max_word_count = word_counts.max()
                if max_word_count <= word_count_threshold:
                    short_text_columns.append(col)
                else:
                    long_text_columns.append(col)

    return numerical_columns, categorical_columns, datetime_columns, short_text_columns, long_text_columns


numerical_columns, categorical_columns, datetime_columns, short_text_columns, long_text_columns = classify_column_types(X)
text_columns = short_text_columns + long_text_columns
print("Numerical columns: ", numerical_columns, "\n")
print("Categorical columns: ", categorical_columns, "\n")
print("DateTime columns:", datetime_columns, "\n")
print("Short text columns:", short_text_columns, "\n")
print("Long text columns:", long_text_columns, "\n")
print("Text columns: ", text_columns, "\n")
print("Data after classifying columns:\n", X.tail(10), "\n")


# Randomly insert duplicates
def insert_duplicates(df, num_duplicates=1000):
    # Ensure the number of duplicates to insert does not exceed the length of the original DataFrame
    num_duplicates = min(num_duplicates, len(df))

    # Randomly select num_duplicates records from df to duplicate
    duplicates = df.sample(n=num_duplicates)

    # Create a new DataFrame that includes both the original data and the duplicates
    df_modified = pd.concat([df, duplicates])

    # Shuffle the rows of the new DataFrame to distribute duplicates randomly
    df_modified = df_modified.sample(frac=1).reset_index(drop=True)

    return df_modified


# Randomly introduce missing values
def introduce_nan(df, numerical_columns, categorical_columns, datetime_columns, text_columns, missing_ratio=0.1):
    df_modified = df.copy()  # Create a copy of the DataFrame to modify

    # Randomly select one column from each type of columns
    selected_num_col = np.random.choice(numerical_columns) if numerical_columns else None
    selected_cat_col = np.random.choice(categorical_columns) if categorical_columns else None
    selected_dt_col = np.random.choice(datetime_columns) if datetime_columns else None
    selected_text_col = np.random.choice(text_columns) if text_columns else None

    for col in [selected_num_col, selected_cat_col, selected_dt_col, selected_text_col]:
        if col is not None:
            total_values = len(df_modified)  # Total number of entries in the column
            existing_missing = df_modified[col].isnull().sum()  # Count existing missing values
            # Calculate the number of new missing values to introduce based on the specified ratio
            new_missing_count = int(total_values * missing_ratio) - existing_missing
            new_missing_count = max(new_missing_count, 0)  # Ensure new_missing_count is non-negative

            if new_missing_count > 0:
                # Get the indices of non-missing values
                non_missing_indices = df_modified[col][df_modified[col].notnull()].index.tolist()
                # Randomly select indices to introduce missing values
                missing_indices = np.random.choice(non_missing_indices, size=new_missing_count, replace=False)
                # Set the selected indices to NaN
                df_modified.loc[missing_indices, col] = np.nan

    return df_modified


# Randomly introduce outliers
def introduce_outliers(df, numerical_columns, num_outliers=1000):
    df_modified = df.copy()

    # Ensure the number of outliers to introduce does not exceed one-tenth of the original DataFrame
    num_outliers = min(num_outliers, len(df_modified)//10)

    for col in numerical_columns:
        # Calculate the max and min values for the column
        col_max = df_modified[col].max()
        col_min = df_modified[col].min()

        # Check if the entire column can be considered of integer type before looping
        is_integer = (df_modified[col].dropna() % 1 == 0).all()

        # Select random indices to introduce outliers
        outlier_indices = np.random.choice(df_modified.index, size=num_outliers, replace=False)
        for idx in outlier_indices:
            # Generate a random factor between 2 and 10
            random_factor = np.random.uniform(2, 10)
            # Randomly decide to set a high or low outlier value
            if np.random.rand() > 0.5:
                df_modified.at[idx, col] = col_max * random_factor  # Multiply the max by a random factor
            else:
                df_modified.at[idx, col] = col_min / random_factor  # Divide the min by a random factor

        # If the column is of integer type, convert the entire column to int after introducing all outliers
        if is_integer:
            df_modified[col] = df_modified[col].apply(lambda x: int(x) if pd.notnull(x) else x).astype('Int64')

    return df_modified


X_random_nans = introduce_nan(X, numerical_columns, categorical_columns, datetime_columns, text_columns)
print("Data with random missing values:\n", X_random_nans.tail(10), "\n")
X_random_outliers = introduce_outliers(X_random_nans, numerical_columns)
print("Data with random outliers:\n", X_random_outliers.tail(10), "\n")
X_random_duplicates = insert_duplicates(X_random_outliers)
print("Data with random duplicates:\n", X_random_duplicates.tail(10), "\n")

X_random_duplicates = X_random_duplicates.dropna(how='all').reset_index(drop=True)

X_distinct = handle_duplicates(X_random_duplicates)
print("Data after handling duplicates:\n", X_distinct.tail(10), "\n")
X_without_outliers = handle_outliers(X_distinct, numerical_columns)
print("Data after handling outliers:\n", X_without_outliers.tail(10), "\n")
X_imputed = handle_missing_values(
    X_without_outliers, numerical_columns, categorical_columns, datetime_columns, short_text_columns, long_text_columns
)
print("Data after imputation:\n", X_imputed.tail(10), "\n")




