print("Selected column for classification: ", selected_col, "\n")

    # Identify columns in modeling_df that start with "selected_col_"
    cols_to_drop = [col for col in modeling_df.columns if col.startswith(f'{selected_col}_')]

    X = modeling_df.drop(columns=cols_to_drop)  # Drop these columns from modeling_df
    y = X_cleaned[selected_col]  # Set y_train to the column from X_cleaned corresponding to selected_col

    # Split X_train and y_train into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # Fit and transform y to have consecutive class labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    # xgb = XGBClassifier()
    #
    # # Define a grid of hyperparameter values for tuning the classifier
    # param_grid = {
    #     'max_depth': [6],  # [2, 4, 6, 8, 10]
    #     'learning_rate': [0.01],  # [0.0001, 0.001, 0.01, 0.1]
    #     'n_estimators': [1000],  # [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    #     'min_child_weight': [1],  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #     'colsample_bytree': [0.7],  # [0.2, 0.4, 0.6, 0.8, 1.0]
    #     'subsample': [0.7],  # [0.2, 0.4, 0.6, 0.8, 1.0]
    #     'reg_alpha': [0.5],  # [0.0, 0.5, 1.0, 5.0, 10.0]
    #     'reg_lambda': [1.0],  # [0.0, 0.5, 1.0, 5.0, 10.0]
    #     'num_parallel_tree': [1],  # [1, 2, 3, 4, 5]
    # }

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
    logger.info(f'Final model:\n {best_model}')
    print("Best parameters found: ", grid_search.best_params_)
    print("Best accuracy found: ", grid_search.best_score_)
    print("Average accuracy: ", grid_search.cv_results_['mean_test_score'].mean())