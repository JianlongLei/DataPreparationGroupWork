from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, NGram, HashingTF, IDF
from pyspark.ml import Pipeline
from timeit import default_timer as timer
import logging

logger = logging.getLogger('dataPreprocessing_Spark')


def encode_categorical(df, categorical_columns):
    logger.info('Started encoding of CATEGORICAL columns...')
    start_time = timer()

    stages = []
    features_created = []  # Initialize a list to track the names of the new features created

    for col in categorical_columns:
        # Count the distinct values in each categorical column
        unique_count = df.agg(F.countDistinct(col).alias("unique_count")).collect()[0]['unique_count']

        # Initialize a StringIndexer for the column
        indexer = StringIndexer(inputCol=col, outputCol=col + "_index").setHandleInvalid("keep")

        # If the number of unique values is 10 or less, use OneHotEncoder
        if unique_count <= 10:
            encoder = OneHotEncoder(inputCols=[indexer.getOutputCol()], outputCols=[col + "_ohe"])
            stages += [indexer, encoder]  # Add both the indexer and encoder to the pipeline stages
            features_created.append(col + "_ohe")
        else:
            # For more than 10 unique values, just use the index
            stages += [indexer]
            features_created.append(col + "_index")

    pipeline = Pipeline(stages=stages)  # Create a Pipeline with the stages defined above
    model = pipeline.fit(df)  # Fit the pipeline to the DataFrame
    df = model.transform(df)  # Transform the DataFrame according to the pipeline

    end_time = timer()
    logger.info(f'Completed encoding of CATEGORICAL columns in {end_time - start_time:.5f} seconds')

    return df, features_created


def convert_datetime(df, datetime_columns):
    logger.info('Started conversion of DATETIME columns...')
    start_time = timer()

    features_created = []

    for col_name in datetime_columns:
        # Get the data type of the current column
        dtype = df.schema[col_name].dataType

        # If the column's data type is either IntegerType or DoubleType, it's a Unix timestamp
        if isinstance(dtype, IntegerType) or isinstance(dtype, DoubleType):
            # Convert Unix timestamp to a timestamp format
            df = df.withColumn(col_name, F.from_unixtime(F.col(col_name)).cast("timestamp"))
        else:
            # Convert the string to a timestamp
            df = df.withColumn(col_name, F.to_timestamp(F.col(col_name)))

        # Extract various datetime components from the timestamp and create new columns for each
        df = df \
            .withColumn(col_name + "_year", F.year(F.col(col_name))) \
            .withColumn(col_name + "_month", F.month(F.col(col_name))) \
            .withColumn(col_name + "_day", F.dayofmonth(F.col(col_name))) \
            .withColumn(col_name + "_weekday", F.weekday(F.col(col_name))) \
            .withColumn(col_name + "_weekday_sin", F.sin(2 * F.pi() * F.col(col_name + "_weekday") / 7)) \
            .withColumn(col_name + "_weekday_cos", F.cos(2 * F.pi() * F.col(col_name + "_weekday") / 7))

        features_created += [col_name + suffix for suffix in
                           ["_year", "_month", "_day", "_weekday", "_weekday_sin", "_weekday_cos"]]

    end_time = timer()
    logger.info(f'Completed conversion of DATETIME columns in {end_time - start_time:.5f} seconds')

    return df, features_created


def extract_text_features(df, text_columns):
    logger.info('Started extraction of features from TEXT columns...')
    start_time = timer()

    features_created = []

    for col in text_columns:
        regexTokenizer = RegexTokenizer(inputCol=col, outputCol="words", pattern="\\W")  # Tokenize the text
        remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")  # Remove stop words
        ngram = NGram(n=2, inputCol="filtered_words", outputCol="ngrams")  # Generate n-grams
        hashingTF = HashingTF(inputCol="ngrams", outputCol="rawFeatures", numFeatures=100)  # Convert n-grams to vectors
        idf = IDF(inputCol="rawFeatures", outputCol=f"{col}_tfidfFeatures")  # Compute TF-IDF

        pipeline = Pipeline(stages=[regexTokenizer, remover, ngram, hashingTF, idf])  # Define the pipeline
        model = pipeline.fit(df)  # Fit the pipeline to the data
        df = model.transform(df)  # Transform the data

        features_created.append(f"{col}_tfidfFeatures")

    # Drop the intermediate columns after all text columns have been processed
    columns_to_drop = ["words", "filtered_words", "ngrams", "rawFeatures"]
    df = df.drop(*columns_to_drop)

    end_time = timer()
    logger.info(f'Completed extraction of features from TEXT columns in {end_time - start_time:.5f} seconds')

    return df, features_created


