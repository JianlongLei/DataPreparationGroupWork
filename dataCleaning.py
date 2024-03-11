import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
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

    # Convert datetime columns to Unix timestamps
    for col in datetime_columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = pd.to_datetime(df[col], unit='s', errors='coerce')
        else:
            df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')

        df[col] = [int(dt.timestamp()) if pd.notnull(dt) else np.nan for dt in df[col]]

    # Loop through numerical, categorical and datetime columns to handle missing values
    for col in (numerical_columns + categorical_columns + datetime_columns):
        if df[col].isnull().any():
            logger.info(f'Processing missing values for column: {col}')

            X = df.copy()  # Create a copy of the DataFrame to avoid modifying the original
            y = X.pop(col)
            X_train = X[y.notnull()]
            y_train = y[y.notnull()]
            X_test = X[y.isnull()]

            transformers = []  # Initialize a list to store transformers

            # Define a transformer for categorical features
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])

            # Append transformers for different types of features, excluding the current target column
            transformers.append(('num', KNNImputer(n_neighbors=3), [c for c in (numerical_columns + datetime_columns) if c != col]))
            transformers.append(('categ', categorical_transformer, [c for c in categorical_columns if c != col]))
            # Add a separate transformer for each text column
            for c in short_text_columns:
                transformers.append((f'text_{c}', TfidfVectorizer(max_features=500), c))
            for c in long_text_columns:
                transformers.append((f'text_{c}', TfidfVectorizer(max_features=1000), c))

            # ColumnTransformer to apply the appropriate transformations to each column type
            feature_transformer = ColumnTransformer(transformers=transformers, remainder="drop")

            # Use predictive modeling for numerical, categorical and datetime columns
            if col in (numerical_columns + datetime_columns):
                pipeline = Pipeline(steps=[
                    ('features', feature_transformer),
                    ('learner', RandomForestRegressor())
                ])
                final_model = pipeline.fit(X_train, y_train)  # Fit the pipeline to the training data
                predicted_values = final_model.predict(X_test)  # Predict missing values for numerical columns
                df.loc[y.isnull(), col] = predicted_values  # Impute the predicted values into the original DataFrame

                if col in datetime_columns:
                    df[col] = pd.to_datetime(df[col], unit='s')

            else:
                le = LabelEncoder()  # Encode labels for categorical target
                encoded_y = le.fit_transform(y_train)

                pipeline = Pipeline(steps=[
                    ('features', feature_transformer),
                    ('learner', RandomForestClassifier())
                ])
                final_model = pipeline.fit(X_train, encoded_y)
                predicted_values = final_model.predict(X_test)
                predicted_values = le.inverse_transform(predicted_values)  # Decode the predictions
                df.loc[y.isnull(), col] = predicted_values

            logger.info(f'Imputed missing values for column: {col}')

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










