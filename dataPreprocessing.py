import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from timeit import default_timer as timer
import logging
logger = logging.getLogger('dataPreprocessing')


# Function for encoding of categorical features in the data
def encode_categorical(df, categorical_columns):
    logger.info('Started encoding of categorical columns...')
    start_time = timer()

    one_hot_encoded_frames = []  # Store DataFrames with one-hot encoded columns to be joined later

    for col in categorical_columns:
        try:
            # Perform OneHot Encoding if the column has 10 or fewer unique values
            if df[col].nunique() <= 10:
                one_hot = pd.get_dummies(df[col], prefix=col)
                one_hot_encoded_frames.append(one_hot)  # Append the one-hot encoded DataFrame to the list

                logger.debug(f'OneHot Encoding succeeded for column "{col}"')
            # Perform Ordinal Encoding if there are more than 10 unique values in the column
            else:
                ordinal_encoder = OrdinalEncoder()
                df[col + '_ordinal'] = ordinal_encoder.fit_transform(df[[col]])

                logger.debug(f'Ordinal Encoding succeeded for column "{col}"')
        except:
            logger.warning(f'Encoding failed for column "{col}"')

    # Join all one-hot encoded frames to the original DataFrame
    if one_hot_encoded_frames:
        df = df.join(one_hot_encoded_frames)

    end_time = timer()
    logger.info(f'Completed encoding of categorical columns in {end_time - start_time:.5f} seconds')

    return df


# Function for extracting of datetime values in the data
def convert_datetime(df, datetime_columns):
    logger.info('Started conversion of DATETIME columns...')
    start_time = timer()

    added_columns = []  # List to keep track of the names of newly added columns
    for col in datetime_columns:
        try:
            # Convert columns with Unix timestamps to datetime format
            if df[col].dtype in ['int64', 'float64']:
                df[col] = pd.to_datetime(df[col], unit='s')
            # For other columns, try to infer the datetime format automatically
            else:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True)

            # Extract and add new columns for year, month, day, and weekday from the datetime column
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_weekday'] = df[col].dt.weekday

            # Perform sinusoidal encoding for the weekday to capture its cyclical nature
            df[f'{col}_weekday_sin'] = np.sin(2 * np.pi * df[f'{col}_weekday'] / 7)
            df[f'{col}_weekday_cos'] = np.cos(2 * np.pi * df[f'{col}_weekday'] / 7)

            # Keep track of all added columns
            added_columns.extend([f'{col}_year', f'{col}_month', f'{col}_day', f'{col}_weekday',
                                  f'{col}_weekday_sin', f'{col}_weekday_cos'])
            logger.debug(f'Conversion to DATETIME succeeded for column "{col}"')

            try:
                # Check if extracted dates are non-NULL; if all are 0, drop the added columns
                if (df[f'{col}_year'] == 0).all() and (df[f'{col}_month'] == 0).all() and (df[f'{col}_day'] == 0).all():
                    for feature in ['year', 'month', 'day', 'weekday', 'weekday_sin', 'weekday_cos']:
                        df.drop(f'{col}_{feature}', inplace=True, axis=1)
                        added_columns.remove(f'{col}_{feature}')
            except:
                pass
        except:
            logger.warning(f'Conversion to DATETIME failed for column "{col}"')

    end_time = timer()
    logger.info(f'Completed conversion of DATETIME columns in {end_time - start_time:.5f} seconds')

    return df, added_columns


# Function for extracting features from text columns in the data
def extract_text_features(df, text_columns):
    # To be implemented
    return df

