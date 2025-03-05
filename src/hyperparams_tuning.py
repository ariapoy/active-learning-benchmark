import os
import json
from xgboost import XGBClassifier
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import LabelEncoder

from utils import *

def tune_and_save_best_params(X_train, y_train, param_grid, dataset_name, cv=3, factor=3):
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
    save_path = f'{dataset_name}-best_params.json'
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
    'n_estimators': [100, 200],
    'max_depth': [0, 1, 3, 6],# [0, 1, 5, 7, 11],
    'min_child_weight':[0, 1, 10], # [0, 1, 10, 50, 100],
    'subsample': [0.5, 0.75, 1],
    'learning_rate': [0.003, 0.03, 0.3], # [0.0003, 0.003, 0.03, 0.3],
    'colsample_bytree': [0.5, 0.75, 1],
    # 'colsample_bylevel': [0.5, 0.7, 0.9, 1],
    'gamma': [3e-3, 3e-1, 0, 3], # [3e-5, 3e-3, 3e-1, 0, 3],
    'lambda': [0, 1, 1.5, 2.5],
    'alpha': [1e-3, 1e-1, 0, 10], # [1e-5, 1e-3, 1e-1, 0, 10],
}

hs_name = 'XGBoost'
data_sets = [
    "appendicitis", "sonar", "parkinsons", "ex8b", "heart", "haberman", "ionosphere", "clean1", "breast", "wdbc", "australian", "diabetes", "mammographic", "ex8a", "tic", "german", "splice", "gcloudb", "gcloudub", "checkerboard",
]

# data_set = 'heart'
for data_set in data_sets:
    data = load_svmlight_file(
        "../data/dataset_used_in_ALSurvey/{0}-svmstyle.txt".format(data_set))
    X, y = data[0], data[1]
    X = np.asarray(X.todense())
    if np.unique(y).shape[0] == 2:  # binary class
        if -1 not in set(y):
            y[y==0] = -1  # bug for hintsvm, dwus?
        if hs_name == 'XGBoost':
            # mapping y to [0, 1, 2, ...]
            y = LabelEncoder().fit_transform(y)
            # TODO address a conflict between XGBoost and hintsvm/dwus
    else:  # multi-class
        # mapping y to [0, 1, 2, ...]
        y = LabelEncoder().fit_transform(y)

    best_params = tune_and_save_best_params(X, y, param_grid, f"{data_set}")

    with open(f'{data_set}-best_params.json', "r") as f:
        best_params_dict = json.load(f)

    model = XGBClassifier(
        **best_params_dict,       # unpack the parameters from the JSON file
        tree_method='hist',    # for efficient training on large datasets
        n_jobs=-1,             # use all available cores
        use_label_encoder=False,
        eval_metric='logloss'
    )

    idx, idx_trn, idx_tst, idx_lbl, idx_ubl = init_data_exps(X, y, 0, 20, 0.4, init_trn_tst='RS', init_trn_tst_fixSeed='noFix', init_lbl_ubl='RS')
    # Get X_trn, X_tst, X_lbl, X_ubl ; y_trn, y_tst, y_lbl, y_ubl
    X_trn, y_trn = X[idx_trn, :], y[idx_trn]
    X_tst, y_tst = X[idx_tst, :], y[idx_tst]
    X_lbl, y_lbl = X[idx_lbl, :], y[idx_lbl]
    model.fit(X_trn, y_trn)
    acc = model.score(X_tst, y_tst)
    with open("best_hyperparams_acc.csv", "a") as g:
        output = f"{data_set},{acc}\n"
        g.write(output)