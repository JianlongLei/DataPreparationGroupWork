from abc import abstractmethod
import tensorflow_data_validation as tfdv


class DataEvaluator:

    @abstractmethod
    def doEvaluate(self, data):
        pass


class TFDVEValuator(DataEvaluator):

    def doEvaluate(self, data):
        # Generate and return a data quality evaluation report
        # This report includes checks for missing values, anomalies, and duplicates
        stats = tfdv.generate_statistics_from_dataframe(data)
        schema = tfdv.infer_schema(stats)
        anomalies = tfdv.validate_statistics(stats, schema)
        return (stats, schema, anomalies)
        
