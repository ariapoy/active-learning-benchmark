from utils import *
import copy
import numpy as np


def select_batch(sampler, uniform_sampler, N, already_selected,
                 **kwargs):
    n_active = int(N)
    n_passive = N - n_active
    kwargs["N"] = n_active
    kwargs["already_selected"] = already_selected
    # issue 2. kcenter. assert ind not in already_selected
    batch_AL = sampler.select_batch(**kwargs)
    already_selected = already_selected + batch_AL
    kwargs["N"] = n_passive
    kwargs["already_selected"] = already_selected
    batch_PL = uniform_sampler.select_batch(**kwargs)
    return batch_AL + batch_PL


def google_al(X_trn, y_trn, X_tst, y_tst, idx_lbl,
              qs, uniform_qs, model_select, model_score, quota,
              **kwargs):
    seed = kwargs['seed']
    # file = open(f'{configs.data_set}-{configs.qs_name}-{configs.hs_name}-{configs.gs_name}-{configs.exp_name}-detail.csv', 'a')

    batch_size = kwargs.get("batch_size")
    if batch_size is None:
        batch_size = 1

    y_all = kwargs.get("y_all")
    indices = kwargs.get("indices")

    # Results
    hist_info = {
        "E_lbl_score": [], "E_trn_score": [], "E_tst_score": [], 'confusion_mat': [], 'idx_qrd_history': idx_lbl.tolist(), 'al_round':[],
        'E_tst_f1score': [],
    }

    # Initialize variables and first select model
    train_size = X_trn.shape[0]
    selected_inds = list(range(idx_lbl.shape[0]))
    # Q1. Why this works?
    # A1. It seems ok as we split X_trn from X_all first.
    X_all = kwargs.get("X_all")
    y_all = kwargs.get("y_all")
    quota_used = 0
    al_round = len(selected_inds)  # num init labelled pool

    partial_X = X_trn[sorted(selected_inds)]
    partial_y = y_trn[sorted(selected_inds)]
    start_train_time = time.time()
    model_select.fit(partial_X, partial_y)
    exec_train_time = time.time() - start_train_time

    # TODO: move evaluation to utils
    model_score.fit(partial_X, partial_y)
    E_trn_score_curr = model_score.score(X_trn, y_trn)
    E_tst_score_curr = model_score.score(X_tst, y_tst)
    hist_info['al_round'].append(al_round)
    hist_info['E_ini_trn_score'] = E_trn_score_curr
    hist_info['E_ini_score'] = E_tst_score_curr
    y_pred = model_score.predict(X_tst)
    confusion_mat_curr = confusion_matrix(y_tst, y_pred).ravel()
    E_tst_f1_score = f1_score(y_tst, y_pred, average='weighted')
    hist_info['confusion_mat_ini'] = confusion_mat_curr
    hist_info['E_tst_f1score'].append(E_tst_f1_score)
    # file.write(f'{seed}|{al_round}|{E_tst_score_curr}|{exec_train_time}|\n')
    logging_print('init', f'|{seed}|{al_round}|{E_tst_score_curr}|{exec_train_time:.3f}|')

    while quota_used < quota:
        # mqke query and get required sample id
        n_sample = min(batch_size, train_size - len(selected_inds))
        select_batch_inputs = {
            "model": model_select,
            "labeled": dict(zip(selected_inds, y_trn[selected_inds])),
            "y": y_trn
        }
        # issue 2. kcenter. assert ind not in already_selected
        start_query_time = time.time()
        new_batch = select_batch(qs, uniform_qs, n_sample,
                                 selected_inds, **select_batch_inputs)
        exec_query_time = time.time() - start_query_time

        # Update lbl, ubl
        selected_inds.extend(new_batch)
        assert len(new_batch) == n_sample
        assert len(list(set(selected_inds))) == len(selected_inds)
        al_round = len(selected_inds)

        # Sort active_ind so that the end results matches that of uniform sampling
        # Q1. Why sorted works?
        # Get y_u = y_ubl[ask_id] from labeler
        partial_X = X_trn[sorted(selected_inds)]
        partial_y = y_trn[sorted(selected_inds)]

        # update model
        start_train_time = time.time()
        model_select.fit(partial_X, partial_y)
        exec_train_time = time.time() - start_train_time

        # eval
        # TODO: move evaluation to utils
        model_score.fit(partial_X, partial_y)
        E_lbl_score_curr = model_score.score(partial_X, partial_y)
        hist_info["E_lbl_score"].append(E_lbl_score_curr)
        E_trn_score_curr = model_score.score(X_trn, y_trn)
        hist_info["E_trn_score"].append(E_trn_score_curr)
        E_tst_score_curr = model_score.score(X_tst, y_tst)
        hist_info["E_tst_score"].append(E_tst_score_curr)
        # confusion matrix at each round, only for test set
        y_pred = model_score.predict(X_tst)
        confusion_mat_curr = confusion_matrix(y_tst, y_pred).ravel()
        hist_info['confusion_mat'].append(confusion_mat_curr)
        E_tst_f1_score = f1_score(y_tst, y_pred, average='weighted')
        hist_info['E_tst_f1score'].append(E_tst_f1_score)
        hist_info['al_round'].append(al_round)

        # file.write(f'{seed}|{al_round}|{E_tst_score_curr}|{exec_train_time}|{exec_query_time}\n')
        logging_print('update', f'|{seed}|{al_round}|{E_tst_score_curr}|{exec_train_time:.3f}|{exec_query_time:.3f}')

        # update used quota
        quota_used += batch_size

    # file.close()
    assert all(y_all[indices[selected_inds]] == y_trn[selected_inds])
    return hist_info
