import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from timeit import default_timer as timer
import logging
logger = logging.getLogger('dataPreprocessing')


# Function for encoding of categorical features in the data
def encode_categorical(df, categorical_columns):
    logger.info('Started encoding of CATEGORICAL columns...')
    start_time = timer()

    features_df = pd.DataFrame(index=df.index)  # Initialize a new DataFrame for added features
    one_hot_encoded_frames = []  # Store DataFrames with one-hot encoded columns to be joined later

    for col in categorical_columns:
        try:
            # Convert column to 'category' data type
            temp_col = df[col].astype('category')

            # Perform OneHot Encoding if the column has 10 or fewer unique values
            if temp_col.nunique() <= 10:
                one_hot = pd.get_dummies(temp_col, prefix=col)
                one_hot_encoded_frames.append(one_hot)  # Append the one-hot encoded DataFrame to the list

                logger.debug(f'OneHot Encoding succeeded for column "{col}"')
            # Perform Label Encoding if there are more than 10 unique values in the column
            else:
                features_df[col + '_label'] = temp_col.cat.codes

                logger.debug(f'Label Encoding succeeded for column "{col}"')
        except:
            logger.warning(f'Encoding failed for column "{col}"')

    # Join all one-hot encoded frames to the new features DataFrame
    if one_hot_encoded_frames:
        features_df = features_df.join(one_hot_encoded_frames)

    end_time = timer()
    logger.info(f'Completed encoding of CATEGORICAL columns in {end_time - start_time:.5f} seconds')

    return features_df


# Function for extracting of datetime values in the data
def convert_datetime(df, datetime_columns):
    logger.info('Started conversion of DATETIME columns...')
    start_time = timer()

    features_df = pd.DataFrame(index=df.index)  # Initialize a new DataFrame for added features
    for col in datetime_columns:
        try:
            # Convert columns with Unix timestamps to datetime format
            if df[col].dtype in ['int64', 'float64']:
                temp_col = pd.to_datetime(df[col], unit='s')
            # For other columns, try to infer the datetime format automatically
            else:
                temp_col = pd.to_datetime(df[col], infer_datetime_format=True)

            # Extract and add new columns for year, month, day, and weekday from the datetime column to features_df
            features_df[f'{col}_year'] = temp_col.dt.year
            features_df[f'{col}_month'] = temp_col.dt.month
            features_df[f'{col}_day'] = temp_col.dt.day
            features_df[f'{col}_weekday'] = temp_col.dt.weekday

            # Perform sinusoidal encoding for the weekday to capture its cyclical nature
            features_df[f'{col}_weekday_sin'] = np.sin(2 * np.pi * features_df[f'{col}_weekday'] / 7)
            features_df[f'{col}_weekday_cos'] = np.cos(2 * np.pi * features_df[f'{col}_weekday'] / 7)

            logger.debug(f'Conversion to DATETIME succeeded for column "{col}"')

            try:
                # Check if extracted dates are non-NULL; if all are 0, plan to drop the added columns
                check_features = [f'{col}_{feature}' for feature in ['year', 'month', 'day']]
                if all((features_df[feature] == 0).all() for feature in check_features):
                    drop_features = [f'{col}_{feature}' for feature in
                                     ['year', 'month', 'day', 'weekday', 'weekday_sin', 'weekday_cos']]
                    features_df.drop(columns=drop_features, inplace=True)
            except:
                pass
        except:
            logger.warning(f'Conversion to DATETIME failed for column "{col}"')

    end_time = timer()
    logger.info(f'Completed conversion of DATETIME columns in {end_time - start_time:.5f} seconds')

    return features_df


# Function for extracting features from text columns in the data
def extract_text_features(df, text_columns):
    logger.info('Started extraction of features from TEXT columns...')
    start_time = timer()

    model = SentenceTransformer("all-MiniLM-L6-v2")  # Load the pre-trained model for text embeddings
    pca = PCA(n_components=100)  # Initialize PCA for dimensionality reduction

    features_df = pd.DataFrame(index=df.index)  # Initialize a DataFrame to hold all new features
    for col in text_columns:
        try:
            # Encode the text data to get the embeddings, and directly convert to float32 to save memory
            embeddings = model.encode(df[col]).astype('float32')

            # Apply PCA on embeddings and reduce memory usage by converting to float32
            reduced_embeddings = pca.fit_transform(embeddings).astype('float32')

            # Create new columns for each principal component in the separate features DataFrame
            for i in range(100):
                features_df[f'{col}_PC{i+1}'] = reduced_embeddings[:, i]

            # Aggregate features based on embeddings and add them to the features DataFrame
            features_df[f'{col}_embedding_mean'] = np.mean(embeddings, axis=1)  # Mean across each dimension
            features_df[f'{col}_embedding_max'] = np.max(embeddings, axis=1)  # Max across each dimension
            features_df[f'{col}_embedding_min'] = np.min(embeddings, axis=1)  # Min across each dimension
            features_df[f'{col}_embedding_std'] = np.std(embeddings, axis=1)  # Standard deviation across each dimension
            features_df[f'{col}_embedding_median'] = np.median(embeddings, axis=1)  # Median across each dimension
            features_df[f'{col}_embedding_l2'] = np.linalg.norm(embeddings, axis=1)  # L2 norm (Euclidean) of each embedding

            logger.debug(f'Feature extraction succeeded for column "{col}"')
        except:
            logger.info(f'ERROR: Feature extraction failed for column "{col}"')
            # logger.warning(f'Feature extraction failed for column "{col}"')

    end_time = timer()
    logger.info(f'Completed extraction of features from TEXT columns in {end_time - start_time:.5f} seconds')

    return features_df



