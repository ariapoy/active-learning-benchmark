from sklearn.base import clone
from utils import *
import numpy as np

def skactiveml_al(X_trn, y_trn_lbl, X_tst, y_tst, y_trn_full,
                  qs, model_select, model_score, quota, batch_size=1,
                  **kwargs):
    # configurations
    y_all = kwargs.get("y_all")
    idx_trn = kwargs.get("idx_trn")
    qs_name = kwargs.get("qs_name")
    configs = kwargs['configs']
    seed = kwargs['seed']
    # quota
    quota_used = 0
    idx_lbl = np.where(~np.isnan(y_trn_lbl))[0]
    al_round = idx_lbl.shape[0]  # num init labelled pool
    # results
    hist_info = {
        "E_lbl_score": [], "E_trn_score": [], "E_tst_score": [], 'confusion_mat': [], 'idx_qrd_history': idx_lbl.tolist()
    }
    # initialize query-oriented and task-oriented models
    start_train_time = time.time()
    model_select.fit(X_trn, y_trn_lbl)
    exec_train_time = time.time() - start_train_time
    # if model_select params is equal to model_score params, then use the same model to accelerate
    # otherwise, use different models to query and score
    is_compatible_models = model_select.estimator.get_params() == model_score.get_params()
    # evaluate by task-oriented model
    X_lbl_curr = X_trn[hist_info['idx_qrd_history']]
    y_lbl_curr = y_trn_full[hist_info['idx_qrd_history']]
    if is_compatible_models:
        E_lbl_score_curr = model_select.score(X_lbl_curr, y_lbl_curr)
        E_trn_score_curr = model_select.score(X_trn, y_trn_full)
        E_tst_score_curr = model_select.score(X_tst, y_tst)
        y_pred = model_select.predict(X_tst)
        confusion_mat_curr = confusion_matrix(y_tst, y_pred).ravel()
    else:
        model_score.fit(X_lbl_curr, y_lbl_curr)
        E_lbl_score_curr = model_score.score(X_lbl_curr, y_lbl_curr)
        E_trn_score_curr = model_score.score(X_trn, y_trn_full)
        E_tst_score_curr = model_score.score(X_tst, y_tst)
        y_pred = model_score.predict(X_tst)
        confusion_mat_curr = confusion_matrix(y_tst, y_pred).ravel()
    
    hist_info['E_ini_score'] = E_tst_score_curr
    hist_info['confusion_mat_ini'] = confusion_mat_curr
    logging_print('init', f'|{seed}|{al_round}|{E_tst_score_curr}|{exec_train_time:.3f}|')

    # active learning loop
    while quota_used < quota:
        # mqke query and get required sample id
        start_query_time = time.time()
        if qs_name == 'skal_bald':
            idx_qrd = qs.query(X=X_trn, y=y_trn_lbl, ensemble=model_select, batch_size=batch_size)
        if qs_name in ['skal_uniform', 'skal_coreset']:
            idx_qrd = qs.query(X=X_trn, y=y_trn_lbl, batch_size=batch_size)
        else:
            idx_qrd = qs.query(X=X_trn, y=y_trn_lbl, clf=model_select)
        exec_query_time = time.time() - start_query_time
        # Update lbl, ubl
        y_trn_lbl[idx_qrd] = y_trn_full[idx_qrd]
        al_round += idx_qrd.shape[0]
        # update model
        start_train_time = time.time()
        model_select.fit(X_trn, y_trn_lbl)
        exec_train_time = time.time() - start_train_time
        # eval
        X_lbl_curr = X_trn[hist_info['idx_qrd_history']]
        y_lbl_curr = y_trn_full[hist_info['idx_qrd_history']]
        if is_compatible_models:
            E_lbl_score_curr = model_select.score(X_lbl_curr, y_lbl_curr)
            E_trn_score_curr = model_select.score(X_trn, y_trn_full)
            E_tst_score_curr = model_select.score(X_tst, y_tst)
            y_pred = model_select.predict(X_tst)
            confusion_mat_curr = confusion_matrix(y_tst, y_pred).ravel()
        else:
            model_score.fit(X_lbl_curr, y_lbl_curr)
            E_lbl_score_curr = model_score.score(X_lbl_curr, y_lbl_curr)
            E_trn_score_curr = model_score.score(X_trn, y_trn_full)
            E_tst_score_curr = model_score.score(X_tst, y_tst)
            y_pred = model_score.predict(X_tst)
            confusion_mat_curr = confusion_matrix(y_tst, y_pred).ravel()
        
        hist_info["E_lbl_score"].append(E_lbl_score_curr)
        hist_info["E_trn_score"].append(E_trn_score_curr)
        hist_info["E_tst_score"].append(E_tst_score_curr)
        hist_info['confusion_mat'].append(confusion_mat_curr)
        hist_info['idx_qrd_history'] += idx_qrd.tolist()
        logging_print('update', f'|{seed}|{al_round}|{E_tst_score_curr}|{exec_train_time:.3f}|{exec_query_time:.3f}')

        # update used quota
        quota_used += batch_size

    assert all(y_all[idx_trn[hist_info['idx_qrd_history']]] == y_trn_full[hist_info['idx_qrd_history']])
    return hist_info