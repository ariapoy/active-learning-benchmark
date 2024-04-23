from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from headers import *

def _logisticRegressionBuilder(**kwargs):
    default_config = {
        'C': 1.0, 'penalty': 'l2', 'dual': False, 'tol': 0.0001, 
        'fit_intercept': True, 'intercept_scaling': 1, 'class_weight': None, 'solver': 'lbfgs', 
        'max_iter': 100, 'multi_class': 'auto', 'verbose': 0, 'warm_start': False, 'n_jobs': None, 
        'l1_ratio': None, 'random_state': 6211
    }
    default_config.update(**kwargs)

    return LogisticRegression(**default_config)

def _SVCBuilder(**kwargs):
    default_config = {
        'kernel': 'rbf', 'gamma': 'auto', 'C': 1.0, 'degree': 3, 'coef0': 0.0, 'shrinking': True, 
        'probability': False, 'tol': 0.001, 'cache_size': 200, 'class_weight': None, 'verbose': False,
        'max_iter': -1, 'decision_function_shape': 'ovr', 'break_ties': False, 'random_state': 6211
    }
    default_config.update(**kwargs)

    return SVC(**default_config)

def _LDABuilder(**kwargs):
    default_config = {
        'solver': 'svd', 'shrinkage': None, 'priors': None, 'n_components': None, 
        'store_covariance': False, 'tol': 0.0001, 'covariance_estimator': None
    }
    default_config.update(**kwargs)

    return LDA(**kwargs)

def _RandomForest(**kwargs):
    default_config = {
        'n_estimators': 100, 'criterion': 'gini', 'max_depth': None, 'min_samples_split': 2,
        'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.0, 'max_features': 'sqrt',
        'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'bootstrap': True,
        'oob_score': False, 'n_jobs': -1, 'random_state': 6211, 'verbose': 0,
        'warm_start': False, 'class_weight': None, 'ccp_alpha': 0.0, 'max_samples': None
    }
    default_config.update(**kwargs)

    return RandomForestClassifier(**kwargs)

def _XGBoostClassifier(**kwargs):
    default_config = {
        'early_stopping_rounds': 2,
    }
    default_config.update(**kwargs)
    return xgb.XGBClassifier(**kwargs)

def SelectModelBuilder(name):
    if name == 'XGBoost':
        return _XGBoostClassifier()
    if name == 'RandomForest':
        return _RandomForest()
    if name == 'RBFSVM':
        return _SVCBuilder()
    if name == 'RBFSVMProb':
        return _SVCBuilder(probability=True)
    if name == 'LR':
        return _logisticRegressionBuilder()
    if name == 'us-zhan':
        return _logisticRegressionBuilder(C=0.1)
    if name == 'qbc-zhan':
        return [
            _logisticRegressionBuilder(),
            _SVCBuilder(kernel='linear', probability=True),
            _SVCBuilder(kernel='rbf', probability=True),
            LDA()
        ]
    if name == 'albl-zhan':
        return _logisticRegressionBuilder()
    if name == 'vr-zhan':
        return _logisticRegressionBuilder()
    if name == 'libact-zhan':
        return None
    if name == 'google-zhan':
        return None
    if name == 'alipy-zhan':
        return None
    if name == 'eer-zhan':
        return _SVCBuilder(probability=True)
    if name == 'bso-zhan':
        return None

    raise NotImplementedError

def ScoreModelBuilder(name):
    if name == 'XGBoost':
        return _XGBoostClassifier()
    if name == 'RandomForest':
        return _RandomForest()
    if name == 'RBFSVMProb':
        return _SVCBuilder(probability=True)
    if name == 'LR':
        return _logisticRegressionBuilder()
    if name == 'LRC=1e-1':
        return _logisticRegressionBuilder(C=0.1)
    if name == 'RBFSVM':
        return _SVCBuilder()
    if 'zhan' in name:
        return _SVCBuilder()

    raise NotImplementedError

def QueryStrategyBuilder(name):
    if name == 'rs':
        return { "qs": libact_RS, "params": {"random_state": 1126 }}
    if name == 'us_lc':
        return { "qs": libact_US, "params": {"model": None, "method": "lc", "random_state": 1126 }}
    if name == 'us_sm':
        return { "qs": libact_US, "params": {"model": None, "method": "sm", "random_state": 1126 }}
    if name == 'us_ent':
        return { "qs": libact_US, "params": {"model": None, "method": "entropy", "random_state": 1126 }}
    if name == 'us-zhan':
        return { "qs": libact_US, "params": {"model": None, "method": "entropy", "random_state": 1126 }}
    if name == 'dwus-zhan':
        return { "qs": libact_DWUS, "params": {"n_clusters": 5, "sigma": 0.1, "max_iter": 100, "tol": 1e-4, "C": 1.0,"kmeans_param": {}, "random_state": 1126 }}
    if name == 'quire-zhan':
        return { "qs": libact_QUIRE, "params": {"lambda": 1.0, "kernel": "rbf", "degree": 3, "gamma": 1.0, "coef0": 1.0 }}
    if name == 'hintsvm':
        return { "qs": libact_HSVM, "params": {"Cl": 1.0, "Ch": 1.0, "p": 0.5, "random_state": 1126, "kernel": "rbf", "degree": 3, "gamma": 0.1, "coef0": 0.0, "tol": 1e-3, "shrinking": 1, "cache_size": 100 }}
    if name == 'hintsvm-zhan':
        return { "qs": libact_HSVM,"params": {"Cl": 1.0, "Ch": 1.0, "p": 0.5, "random_state": 1126, "kernel": "linear", "degree": 3, "gamma": 0.1, "coef0": 0.0, "tol": 1e-3, "shrinking": 1, "cache_size": 100 }}
    if name == 'qbc-zhan':
        return { "qs": libact_QBC,"params": {"models": [], "disagreement": "vote", "random_state": 1126 }}
    if name == 'albl-zhan':
        return { "qs": libact_ALBL,"params": {"T": None,
            "query_strategies": [
                functools.partial(libact_US, ds=None,
                    model=libact_skProbaAdapter(_logisticRegressionBuilder(C=1)),
                    random_state=1126
                    ),
                functools.partial(libact_US, ds=None,
                    model=libact_skProbaAdapter(_logisticRegressionBuilder(C=.01)),
                    random_state=1126
                    ),
                functools.partial(libact_HSVM, ds=None, random_state=1126)
                ],
            "delta": 0.1, "uniform_sampler": True, "model": None, "random_state": 1126 }}

    if name == 'uniform-zhan':
        return { "qs": AL_MAPPING["uniform"], "params": {"seed": 1126 }}
    if name == 'libactUniform-zhan':
        return { "qs": libact_RS, "params": {"random_state": 1126 }}
    if name == 'alipyUniform-zhan':
        return { "qs": QueryInstanceRandom, "params": {"seed": 1126 }, "select": {"batch_size": 1}}
    if name == 'margin-zhan':
        return { "qs": AL_MAPPING["margin"], "params": {"seed": 1126 }}
    if name == 'graph-zhan':
        return { "qs": AL_MAPPING["graph_density"], "params": {"seed": 1126 }}
    if name == 'hier-zhan':
        return { "qs": AL_MAPPING["hierarchical"],"params": {"seed": 1126, "beta": 2, "affinity": 'euclidean', "linkage": 'ward', "clustering": None, "max_features": None }}
    if name == 'infodiv-zhan':
        return { "qs": AL_MAPPING["informative_diverse"], "params": {"seed": 1126} }
    if name == 'mcm-zhan':
        return { "qs": AL_MAPPING["margin_cluster_mean"], "params": {"seed": 1126} }
    if name == 'usalipy-zhan':
        return { "qs": QueryInstanceUncertainty, "params": {"measure": 'entropy'}, "select": {"model": None, "batch_size": 1} }
    if name == 'eer-zhan':
        return { "qs": QueryExpectedErrorReduction, "params": {}, "select": {"model": None, "batch_size": 1} }
    if name == 'bmdr-zhan':
        return { "qs": QueryInstanceBMDR, "params": { "beta": 1000, "gamma": 0.1, "rho": 1, "kernel": "rbf", "degree": 3, "gamma_ker": 1, "coef0": 1 }, "select": {"batch_size": 1, "qp_solver": 'ECOS'} }
    if name == 'spal-zhan':
        return { "qs": QueryInstanceSPAL, "params": { "mu": 0.1, "gamma": 0.1, "rho": 1, "lambda_init": 0.1, "lambda_pace": 0.01, "kernel": "rbf", "degree": 3, "gamma_ker": 1, "coef0": 1 }, "select": {"batch_size": 1, "qp_solver": 'ECOS'} }
    if name == 'lal-zhan':
        return { "qs": QueryInstanceLAL, "params": { "mode": "LAL_iterative", "data_path": "alipy-log/", "cls_est": 50, "train_slt": True }, "select": {"batch_size": 1} }
    if name == 'bso-zhan':
        return { "qs": None, "params": {"num_beam": 5} }
    if name == 'vr-zhan':
        return { "qs": libact_VR, "params": {"model": None, "sigma": 100.0, "optimality": "trace", "n_jobs": 1}, }
    if name == 'kcenter-zhan':
        return { "qs": libactKCG, "params": {"seed": 1126, "metric": 'euclidean'} }
    if name == 'skactiveml_margin':
        return { 'qs': UncertaintySampling, 'params': {'method': 'margin_sampling', 'random_state': 1126} }
    if name == 'skactiveml_bald':
        return { 'qs': BatchBALD, 'params': {'random_state': 1126} }

    raise NotImplementedError
