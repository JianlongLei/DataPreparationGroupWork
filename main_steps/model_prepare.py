from abc import abstractmethod
from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from algorithms.dataPreprocessing import encode_categorical, convert_datetime, extract_text_features


class ModelPrepare:
    def __init__(self,
                 train_data: pd.DataFrame,
                 train_labels: pd.Series,
                 test_data: pd.DataFrame,
                 test_labels: pd.Series,
                 total_data: pd.DataFrame,
                 categorical_columns: List[str] = [],
                 numerical_columns: List[str] = [],
                 text_columns: List[str] = [],
                 datetime_columns: List[str] = []
                 ):
        self.model = RandomForestRegressor(n_estimators=100, max_depth=30)
        self._baseline_model = None

        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.total_data = total_data
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.datetime_columns = datetime_columns
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


class NewModelPrepare(ModelPrepare):

    def get_params(self, transformers: ColumnTransformer):
        return super().get_params(transformers)

    def new_fit(self, selected_col):
        print("Selected column for classification: ", selected_col, "\n")

        categorical_features_df = encode_categorical(self.total_data, self.categorical_columns)
        datetime_features_df = convert_datetime(self.total_data, self.datetime_columns)
        text_features_df = extract_text_features(self.total_data, self.text_columns)

        modeling_df = pd.concat([self.total_data[self.numerical_columns], categorical_features_df,
                                 datetime_features_df, text_features_df], axis=1)
        # Identify columns in modeling_df that start with "selected_col_"
        cols_to_drop = [col for col in modeling_df.columns if col.startswith(f'{selected_col}_')]

        X = modeling_df.drop(columns=cols_to_drop)  # Drop these columns from modeling_df
        y = self.total_data[selected_col]  # Set y_train to the column from X_cleaned corresponding to selected_col

        # Split X_train and y_train into training and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

        # Fit and transform y to have consecutive class labels
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)

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

        # Fit the model on the training split
        grid_search.fit(X_train, y_train)

        # Extract the best model
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)  # Predict on the test data
        y_pred = le.inverse_transform(y_pred)  # Decode the predictions

        # Calculate the accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Print out the results
        print("Test data accuracy: ", accuracy)
        print("Best parameters found: ", grid_search.best_params_)
        print("Best accuracy found: ", grid_search.best_score_)
        print("Average accuracy: ", grid_search.cv_results_['mean_test_score'].mean())
        return accuracy
