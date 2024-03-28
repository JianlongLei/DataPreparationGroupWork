from abc import abstractmethod
from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    f1_score,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score
)


class ModelPrepare:
    def __init__(self,
                 train_data: pd.DataFrame,
                 train_labels: pd.Series,
                 test_data: pd.DataFrame,
                 test_labels: pd.Series,
                 categorical_columns: List[str] = [],
                 numerical_columns: List[str] = [],
                 text_columns: List[str] = [],
                 ):
        self.model = RandomForestRegressor(n_estimators=100, max_depth=30)
        self._baseline_model = None

        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.text_columns = text_columns

    @abstractmethod
    def get_params(self, transformers: ColumnTransformer):
        """
        Forces child class to define task specific `Pipeline`, hyperparameter grid for HPO, and scorer for baseline model training.
        This helps to reduce redundant code.

        Args:
            transformers (ColumnTransformer): Basic preprocessing for columns. Given by `fit_baseline_model` that calls this method

        Returns:
            Tuple[Dict[str, object], Any, Dict[str, Any]]: Task specific parts to build baseline model
        """
        pass

    def fit(self):
        transformers = []
        if len(self.numerical_columns) > 0:
            transformers.append(('numerical_features', StandardScaler(), self.numerical_columns))
        if len(self.categorical_columns) > 0:
            transformers.append(
                ('categorical_features', OneHotEncoder(handle_unknown='ignore'), self.categorical_columns))
        if len(self.text_columns) > 0:
            for index, text_name in enumerate(self.text_columns):
                transformers.append((f'textual_features_{index}', TfidfVectorizer(), text_name))
        feature_transformation = ColumnTransformer(transformers=transformers, remainder="drop")

        # param_grid, pipeline, scorer = self.get_params(feature_transformation)
        param_grid, pipeline = self.get_params(feature_transformation)
        # refit = list(scorer.keys())[0]

        search = GridSearchCV(pipeline, param_grid, cv=5)
        train_features = self.train_data[self.numerical_columns + self.categorical_columns + self.text_columns]
        model = search.fit(train_features, self.train_labels).best_estimator_

        return model


class RandomForestModelPrepare(ModelPrepare):

    def get_params(self, transformers: ColumnTransformer):
        param_grid = {
            # 'classifier__n_estimators': [100, 150, 200],
            # 'classifier__max_depth': [10, 20, 30],
            'classifier__n_estimators': [50],
            'classifier__max_depth': [10],
        }

        pipeline = Pipeline(
            [
                ('features', transformers),
                ('classifier', RandomForestClassifier())
            ]
        )

        # scorer = {
        #     "ROC/AUC": make_scorer(roc_auc_score, needs_proba=True)
        # }

        # return param_grid, pipeline, scorer
        return param_grid, pipeline

