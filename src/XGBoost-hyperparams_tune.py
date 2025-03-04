import os
import json
from xgboost import XGBClassifier
from sklearn.model_selection import HalvingGridSearchCV

def tune_and_save_best_params(X_train, y_train, param_grid, dataset_name, save_path="best_params.json", cv=3, factor=3):
    """
    Tune XGBoost hyper-parameters using HalvingGridSearchCV for a given dataset
    and save the best parameters to a JSON file.
    
    Parameters:
        X_train (array-like): Feature matrix for training.
        y_train (array-like): Target vector.
        param_grid (dict): Dictionary where keys are parameter names and values are lists of candidate values.
        dataset_name (str): Unique identifier for the dataset.
        save_path (str, optional): File path to store best hyper-parameters (default "best_params.json").
        cv (int, optional): Number of folds in cross-validation (default 3).
        factor (int, optional): The reduction factor at each iteration (default 3).
    
    Returns:
        dict: Best hyper-parameters found for the dataset.
    """
    
    # Initialize the XGBClassifier with efficient settings:
    # Using tree_method='hist' speeds up training on large datasets.
    xgb_clf = XGBClassifier(
        tree_method='hist',
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Set up HalvingGridSearchCV for an efficient hyperparameter search
    search = HalvingGridSearchCV(
        estimator=xgb_clf,
        param_grid=param_grid,
        cv=cv,
        factor=factor,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the model to the training data
    search.fit(X_train, y_train)
    best_params = search.best_params_
    
    # Load existing best parameters from file (if any)
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            all_best_params = json.load(f)
    else:
        all_best_params = {}
    
    # Update dictionary with best parameters for the current dataset
    all_best_params[dataset_name] = best_params
    
    # Save the updated dictionary back to file
    with open(save_path, 'w') as f:
        json.dump(all_best_params, f, indent=4)
    
    print(f"Best parameters for dataset '{dataset_name}': {best_params}")
    return best_params

# Example usage:
# Define a parameter grid for tuning
param_grid = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2]
}

# Suppose you have a dataset (X_train, y_train) and a unique name "dataset_A"
# best_params = tune_and_save_best_params(X_train, y_train, param_grid, "dataset_A")