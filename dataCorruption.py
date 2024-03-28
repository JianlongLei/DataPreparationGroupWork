# Randomly insert duplicates
import numpy as np
import pandas as pd


def insert_duplicates(df):
    # Add a column to preserve the original order of the dataframe
    df['original_order'] = range(len(df))

    num_duplicates = np.random.randint(1, 10001)
    num_duplicates = min(num_duplicates, len(df))  # Ensure the number of duplicates does not exceed the original size
    duplicates = df.sample(n=num_duplicates)  # Randomly select records to duplicate

    # Assign new 'original_order' positions for duplicates to insert them at random positions
    duplicates['original_order'] = np.random.choice(df['original_order'], size=num_duplicates, replace=False)

    df_modified = pd.concat([df, duplicates])

    # Sort the modified dataframe by 'original_order' to mix in the duplicates at their new positions
    df_modified = df_modified.sort_values(by='original_order').reset_index(drop=True)
    df_modified.drop(columns=['original_order'], inplace=True)  # Drop the 'original_order' column
    df.drop(columns=['original_order'], inplace=True)  # Drop the 'original_order' column from the original DataFrame

    return df_modified


# Randomly introduce missing values
def introduce_nan(df, numerical_columns, categorical_columns, datetime_columns, text_columns, missing_ratio=0.02):
    df_modified = df.copy()  # Create a copy of the DataFrame to modify

    # Randomly select one column from each type of columns
    selected_num_col = np.random.choice(numerical_columns) if numerical_columns else None
    selected_cat_col = np.random.choice(categorical_columns) if categorical_columns else None
    selected_dt_col = np.random.choice(datetime_columns) if datetime_columns else None
    selected_text_col = np.random.choice(text_columns) if text_columns else None

    for col in [selected_num_col, selected_cat_col, selected_dt_col, selected_text_col]:
        if col is not None and col in df_modified.columns:
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
                originType = df_modified[col].dtype
                if originType in ['int64', 'float64']:
                    df_modified.loc[missing_indices, col] = np.nan
                else:
                    df_modified.loc[missing_indices, col] = pd.NA
                    df_modified[col] = df_modified[col].astype(originType)


    return df_modified


# Randomly introduce outliers
def introduce_outliers(df, numerical_columns):
    df_modified = df.copy()

    # Ensure the number of outliers to introduce does not exceed one-tenth of the original DataFrame
    num_outliers = np.random.randint(1, 10001)
    num_outliers = min(num_outliers, len(df_modified)//10)

    for col in numerical_columns:
        # Calculate the max and min values for the column
        col_max = df_modified[col].max()
        col_min = df_modified[col].min()

        # Check if the entire column can be considered of integer type before looping
        # is_integer = (df_modified[col].dropna() % 1 == 0).all()

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
        # if is_integer:
        #     df_modified[col] = df_modified[col].apply(lambda x: int(x) if pd.notnull(x) else x).astype('Int64')

    return df_modified
