from dataCleaning import *
from dataCorruption import introduce_nan, introduce_outliers, insert_duplicates
from dataPreprocessing import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import logging
import warnings
import json

logging.basicConfig(level=logging.DEBUG,
                    filename='Data_Cleaning_and_Preprocessing.log',  # log file
                    filemode='w',  # write mode
                    format='%(asctime)s - %(levelname)s - %(message)s')  # log format
warnings.filterwarnings('ignore')

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
                if sample_data.between(1e9, 2e9).any() and "time" in col.lower():
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


# Data corruption
X_random_nans = introduce_nan(X, numerical_columns, categorical_columns, datetime_columns, text_columns)
X_random_outliers = introduce_outliers(X_random_nans, numerical_columns)
X_random_duplicates = insert_duplicates(X_random_outliers)
print("Corrupted data:\n", X_random_duplicates.tail(10), "\n")


# Data cleaning
X_random_duplicates = X_random_duplicates.dropna(how='all').reset_index(drop=True)

X_distinct = handle_duplicates(X_random_duplicates)
print("Data after handling duplicates:\n", X_distinct.tail(10), "\n")
X_without_outliers = handle_outliers(X_distinct, numerical_columns)
print("Data after handling outliers:\n", X_without_outliers.tail(10), "\n")
X_cleaned = handle_missing_values(X_without_outliers, numerical_columns, categorical_columns, datetime_columns,
                                  short_text_columns, long_text_columns)
print("Data after cleaning:\n", X_cleaned.tail(10), "\n")


# Data preprocessing after cleaning
categorical_features_df = encode_categorical(X_cleaned, categorical_columns)
print("Categorical features encoded:\n", categorical_features_df.tail(10), "\n")
datetime_features_df = convert_datetime(X_cleaned, datetime_columns)
print("DateTime features extracted:\n", datetime_features_df.tail(10), "\n")
text_features_df = extract_text_features(X_cleaned, text_columns)
print("Text features extracted:\n", text_features_df.tail(10), "\n")

# Concatenate all DataFrames horizontally (axis=1) to form a complete feature set
modeling_df = pd.concat([X_cleaned[numerical_columns], categorical_features_df,
                         datetime_features_df, text_features_df], axis=1)
print("Data used for modeling:\n", modeling_df.tail(10), "\n")


# Model training and evaluation
logger.info('Started model training and evaluation...')
start_time = timer()
selected_col = np.random.choice(categorical_columns) if categorical_columns else None
if selected_col is not None:
    print("Selected column for classification: ", selected_col, "\n")

    # Identify columns in modeling_df that start with "selected_col_"
    cols_to_drop = [col for col in modeling_df.columns if col.startswith(f'{selected_col}_')]

    X_train = modeling_df.drop(columns=cols_to_drop)  # Drop these columns from modeling_df
    y_train = X_cleaned[selected_col]  # Set y_train to the column from X_cleaned corresponding to selected_col

    # Fit and transform y to have consecutive class labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    # xgb = XGBClassifier()
    #
    # # Define a grid of hyperparameter values for tuning the classifier
    # param_grid = {
    #     'max_depth': [6],  # [2, 4, 6, 8, 10]
    #     'learning_rate': [0.01],  # [0.0001, 0.001, 0.01, 0.1]
    #     'n_estimators': [1000],  # [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    #     'min_child_weight': [1],  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #     'colsample_bytree': [0.7],  # [0.2, 0.4, 0.6, 0.8, 1.0]
    #     'subsample': [0.7],  # [0.2, 0.4, 0.6, 0.8, 1.0]
    #     'reg_alpha': [0.5],  # [0.0, 0.5, 1.0, 5.0, 10.0]
    #     'reg_lambda': [1.0],  # [0.0, 0.5, 1.0, 5.0, 10.0]
    #     'num_parallel_tree': [1],  # [1, 2, 3, 4, 5]
    # }

    clf = RandomForestClassifier()

    # Define a grid of hyperparameter values for tuning the classifier
    param_grid = {
        'n_estimators': [200],  # [100, 200, 300, 400, 500]
        'max_depth': [None],  # [None, 10, 20, 30, 40, 50]
        'min_samples_split': [2],  # [2, 5, 10]
        'min_samples_leaf': [1],  # [1, 2, 4]
        'max_features': ['sqrt'],  # ['auto', 'sqrt', 'log2']
        'bootstrap': [True],  # [True, False]
        'criterion': ['gini'],  # ['gini', 'entropy']
    }

    # Set up GridSearchCV to find the best model parameters using 5-fold cross-validation
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')

    grid_search.fit(X_train, y_train)

    # Extract the best model
    best_model = grid_search.best_estimator_  # the best estimator (the trained model with the best parameters)
    logger.info(f'Final model:\n {best_model}')

    # Access the best parameters and the best score after fitting
    print("Best parameters found: ", grid_search.best_params_)
    print("Best accuracy found: ", grid_search.best_score_)
    print("Average accuracy: ", grid_search.cv_results_['mean_test_score'].mean())

end_time = timer()
logger.info(f'Model training and evaluation completed in {end_time - start_time:.5f} seconds')






