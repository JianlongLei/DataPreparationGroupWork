from abc import abstractmethod
from pyspark.sql import SparkSession
import pydeequ

from evaluation.Deequ_DataQualityEvaluator import DataQualityEvaluator


class DataEvaluator:

    def __init__(self, data=None):
        self.data = data

    def setData(self, data):
        self.data = data

    @abstractmethod
    def doEvaluate(self):
        pass


class DeequEvaluator(DataEvaluator):

    def __init__(self, data=None):
        super().__init__(data)
        # Create a SparkSession
        self.spark = ((SparkSession.builder
                       .appName("DeequEvaluator"))
                      .config("spark.jars.packages", pydeequ.deequ_maven_coord)
                      .config("spark.jars.excludes", pydeequ.f2j_maven_coord)
                      .getOrCreate())

        # Create DataQualityEvaluator
        self.evaluator = DataQualityEvaluator(self.spark)

    def doEvaluate(self):
        # Validate data
        validation_results = self.evaluator.validate_data(self.data)

        # Print the results
        for result in validation_results:
            print(result)

    def stopSession(self):
        # Stop SparkSession
        self.spark.stop()
