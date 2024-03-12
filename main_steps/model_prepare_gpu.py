from cuml.preprocessing import StandardScaler
from cuml.preprocessing import OneHotEncoder
from cuml.ensemble import RandomForestClassifier
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.compose import ColumnTransformer
from cuml.pipeline import Pipeline
from cuml.model_selection import GridSearchCV

from model_prepare import ModelPrepare


class GPUClassificationPrepare(ModelPrepare):

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
        model = search.fit(self.train_data, self.train_labels).best_estimator_

        return model


    def get_params(self, transformers: ColumnTransformer):
        param_grid = {
            'classifier__n_estimators': [100, 150, 200],
            'classifier__max_depth': [10, 20, 30],
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