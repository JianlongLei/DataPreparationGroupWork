import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from dataPreprocessing import convert_datetime
from timeit import default_timer as timer
import logging
logger = logging.getLogger('dataCleaning')


def handle_duplicates(df):
    logger.info('Started handling of duplicates...')
    start_time = timer()
    # Record the number of rows before handling duplicates
    rows_before = df.shape[0]
    try:
        # Remove duplicate rows and reset index
        df = df.drop_duplicates().reset_index(drop=True)
        rows_after = df.shape[0]
        num_duplicates = rows_before - rows_after
        # Log the outcome of duplicates handling
        if num_duplicates > 0:
            logger.debug(f'Deletion of {num_duplicates} duplicate(s) succeeded')
        else:
            logger.debug('No duplicates found')
        end_time = timer()
        logger.info(f'Handling of duplicates completed in {end_time - start_time:.5f} seconds')
    except:
        logger.warning('Handling of duplicates failed')

    return df


def handle_missing_values(df, numerical_columns, categorical_columns, datetime_columns, short_text_columns, long_text_columns):
    logger.info('Started handling of missing values...')
    start_time = timer()
    num_missing = df.isnull().sum().sum()  # Calculate the total number of missing values
    logger.info(f'Found a total of {num_missing} missing value(s)')

    # Fill missing values in text columns with an empty string immediately
    for col in (short_text_columns + long_text_columns):
        if df[col].isnull().any():  # Check if the column has any missing values
            logger.info(f'Processing missing values for column: {col}')
            df[col] = df[col].fillna('')
            logger.info(f'Imputed missing values for column: {col}')

    for col in datetime_columns:
        if df[col].isnull().any():
            logger.info(f'Processing missing values for column: {col}')
            if df[col].dtype in ['int64', 'float64']:
                df[col] = pd.to_datetime(df[col], unit='s', errors='coerce')
            else:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')

            # df[col].interpolate(method='linear', inplace=True)  # Fill missing values using linear interpolation

            timestamps = df[col].dropna().astype(np.int64) // 10 ** 9  # Convert datetime to timestamp
            median_timestamp = timestamps.median()  # Calculate the median datetime
            median_datetime = pd.to_datetime(median_timestamp, unit='s')  # Convert the median timestamp back to datetime
            df[col].fillna(median_datetime, inplace=True)  # Fill missing values with the median datetime

            logger.info(f'Imputed missing values for column: {col}')

    datetime_features_df = convert_datetime(df, datetime_columns)
    datetime_features = datetime_features_df.columns.tolist()
    df = df.join(datetime_features_df)

    # Loop through numerical and categorical columns to handle missing values
    for col in (numerical_columns + categorical_columns):
        if df[col].isnull().any():
            logger.info(f'Processing missing values for column: {col}')

            X = df.copy()  # Create a copy of the DataFrame to avoid modifying the original
            y = X.pop(col)
            X_train = X[y.notnull()]
            y_train = y[y.notnull()]
            X_test = X[y.isnull()]

            transformers = []  # Initialize a list to store transformers

            # Define a transformer for numerical features (imputation and scaling)
            numerical_transformer = Pipeline(steps=[
                ('imputer', KNNImputer(n_neighbors=3)),
                ('scaler', StandardScaler())
            ])

            # Define a transformer for categorical features (imputation and encoding)
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])

            # Define a function to convert numpy ndarray to a flat list of strings
            def to_flat_list(ndarray):
                return ndarray.flatten().astype(str).tolist()

            # Transformer to convert ndarray to a flat list for text processing
            ndarray_to_list_transformer = FunctionTransformer(to_flat_list, validate=False)

            # Define transformers for text features (imputation and vectorization)
            short_text_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='')),
                ('ndarray_to_list', ndarray_to_list_transformer),
                ('vectorizer', TfidfVectorizer(max_features=500))
            ])

            long_text_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='')),
                ('ndarray_to_list', ndarray_to_list_transformer),
                ('vectorizer', TfidfVectorizer(max_features=1000))
            ])

            # Append transformers for different types of features, excluding the current target column
            transformers.append(('num', numerical_transformer, [c for c in numerical_columns if c != col] + datetime_features))
            transformers.append(('categ', categorical_transformer, [c for c in categorical_columns if c != col]))
            # Add a separate transformer for each text column
            for c in short_text_columns:
                transformers.append((f'text_{c}', short_text_transformer, [c]))
            for c in long_text_columns:
                transformers.append((f'text_{c}', long_text_transformer, [c]))

            # ColumnTransformer to apply the appropriate transformations to each column type
            feature_transformer = ColumnTransformer(transformers=transformers, remainder="drop")

            # Use predictive modeling for numerical and categorical columns
            if col in numerical_columns:
                pipeline = Pipeline(steps=[
                    ('features', feature_transformer),
                    ('learner', RandomForestRegressor())
                ])
                final_model = pipeline.fit(X_train, y_train)  # Fit the pipeline to the training data
                predicted_values = final_model.predict(X_test)  # Predict missing values for numerical columns
            else:
                label_encoder = LabelEncoder()  # Encode labels for categorical target
                encoded_y = label_encoder.fit_transform(y_train)

                pipeline = Pipeline(steps=[
                    ('features', feature_transformer),
                    ('learner', RandomForestClassifier())
                ])
                # Fit the pipeline to the training data
                final_model = pipeline.fit(X_train, encoded_y)
                encoded_predictions = final_model.predict(X_test)
                predicted_values = label_encoder.inverse_transform(encoded_predictions)  # Decode the predictions

            df.loc[y.isnull(), col] = predicted_values  # Impute the predicted values into the original DataFrame

            logger.info(f'Imputed missing values for column: {col}')

    df.drop(datetime_features, axis=1, inplace=True)  # Drop the added columns

    end_time = timer()
    logger.info(f'Handling of missing values completed in {end_time - start_time:.5f} seconds')

    return df


# Function for outlier winsorization
def handle_outliers(df, numerical_columns):
    logger.info('Started handling of outliers...')
    start_time = timer()

    for col in numerical_columns:
        # Calculate the bounds for identifying outliers
        outlier_param = 1.5
        q1, q3 = np.percentile(df[col].dropna(), [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (outlier_param * iqr)
        upper_bound = q3 + (outlier_param * iqr)

        # Check if the entire column can be considered of integer type before looping
        is_integer = (df[col].dropna() % 1 == 0).all()

        counter = 0  # Initialize a counter to track the number of outliers handled
        for row_index, row_val in enumerate(df[col]):
            # Skip NaN values
            if pd.isna(row_val):
                continue

            # Check if the value is an outlier (outside the bounds)
            if row_val < lower_bound or row_val > upper_bound:
                # Replace outliers with the corresponding bound
                df.at[row_index, col] = lower_bound if row_val < lower_bound else upper_bound
                counter += 1

        # If the column is of integer type, convert the entire column to int after handling all outliers
        if is_integer:
            df[col] = df[col].apply(lambda x: int(x) if pd.notnull(x) else x).astype('Int64')

        if counter > 0:
            logger.debug(f'Outlier imputation of {counter} value(s) succeeded for numerical column "{col}"')

    end_time = timer()
    logger.info(f'Handling of outliers completed in {end_time - start_time:.5f} seconds')

    return df










