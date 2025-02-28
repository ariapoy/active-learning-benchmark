from utils import *
import copy
import numpy as np


def libact_al(trn_ds, tst_ds, openmap_trn_ds, qs, model_select, model_score, quota, lbr, **kwargs):
    configs = kwargs['configs']
    seed = kwargs['seed']
    # file = open(f'{configs.data_set}-{configs.qs_name}-{configs.hs_name}-{configs.gs_name}-{configs.exp_name}-detail.csv', 'a')
    # Results
    hist_info = {
        "E_lbl_score": [], "E_trn_score": [], "E_tst_score": [], 'confusion_mat': [], 'al_round':[],
    }
    # quota
    quota_used = 0
    al_round = trn_ds.len_labeled()  # num init labelled pool

    # init model
    start_train_time = time.time()
    qs.model = model_select[0]
    qs.model.train(trn_ds)
    exec_train_time = time.time() - start_train_time

    # evaluation by score model
    X_tst_libact, y_tst_libact = tst_ds._X, tst_ds._y
    X_trn_libact, y_trn_libact = openmap_trn_ds._X, openmap_trn_ds._y
    model_score_libact = copy.deepcopy(model_score)
    trn_curr = trn_ds.get_labeled_entries()
    trn_X_curr = trn_curr[0]
    trn_y_curr = np.asarray(trn_curr[1])  # , dtype=int
    # update init score model by init labelled pool
    model_score_libact.fit(trn_X_curr, trn_y_curr)

    # TODO: move evaluation to utils
    E_lbl_score_curr = model_score_libact.score(trn_X_curr, trn_y_curr)
    E_trn_score_curr = model_score_libact.score(X_trn_libact, y_trn_libact)
    E_tst_score_curr = model_score_libact.score(X_tst_libact, y_tst_libact)
    y_pred = model_score_libact.predict(X_tst_libact)
    confusion_mat_curr = confusion_matrix(y_tst_libact, y_pred).ravel()
    hist_info['al_round'].append(al_round)
    hist_info['E_ini_trn_score'] = E_trn_score_curr
    hist_info['E_ini_score'] = E_tst_score_curr
    hist_info['confusion_mat_ini'] = confusion_mat_curr
    # file.write(f'{seed}|{al_round}|{E_tst_score_curr}|{exec_train_time}|\n')
    logging_print('init', f'|{seed}|{al_round}|{E_tst_score_curr}|{exec_train_time:.3f}|')

    while quota_used < quota:
        # mqke query and get required sample id
        start_query_time = time.time()
        ask_id = qs.make_query()
        X, _ = zip(*trn_ds.data)
        if isinstance(ask_id, list):  # libact==1.0, for batch mode in the future
            ask_id = ask_id[0]
        exec_query_time = time.time() - start_query_time
        # Get y_u = y_ubl[ask_id] from labeler
        y_u = lbr.label(X[ask_id])
        # Update lbl, ubl
        trn_ds.update(ask_id, y_u)

        al_round = trn_ds.len_labeled()

        # evaluation by score model
        # retrain from original model
        model_score_libact = copy.deepcopy(model_score)
        # get data
        X_tst_libact, y_tst_libact = tst_ds._X, tst_ds._y
        trn_curr = trn_ds.get_labeled_entries()
        trn_X_curr = trn_curr[0]
        trn_y_curr = np.asarray(trn_curr[1])
        X_trn_libact, y_trn_libact = openmap_trn_ds._X, openmap_trn_ds._y
        # update model
        start_train_time = time.time()
        model_score_libact.fit(trn_X_curr, trn_y_curr)
        exec_train_time = time.time() - start_train_time

        # evaluation
        # TODO: move evaluation to utils
        E_lbl_score_curr = model_score_libact.score(trn_X_curr, trn_y_curr)
        hist_info["E_lbl_score"].append(E_lbl_score_curr)
        E_trn_score_curr = model_score_libact.score(X_trn_libact, y_trn_libact)
        hist_info["E_trn_score"].append(E_trn_score_curr)
        E_tst_score_curr = model_score_libact.score(X_tst_libact, y_tst_libact)
        hist_info["E_tst_score"].append(E_tst_score_curr)
        # confusion matrix at each round, only for test set
        y_pred = model_score_libact.predict(X_tst_libact)
        confusion_mat_curr = confusion_matrix(y_tst_libact, y_pred).ravel()
        hist_info['confusion_mat'].append(confusion_mat_curr)
        hist_info['al_round'].append(al_round)

        # file.write(f'{seed}|{al_round}|{E_tst_score_curr}|{exec_train_time}|{exec_query_time}\n')
        logging_print('update', f'|{seed}|{al_round}|{E_tst_score_curr}|{exec_train_time:.3f}|{exec_query_time:.3f}')
        # update used quota
        quota_used += 1

    # file.close()
    return hist_info
