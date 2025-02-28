from utils import *
import copy
import numpy as np
import functools
sys.path.append("../../alipy-dev/")
from alipy_dev.experiment import State

def alipy_al_exps(qs, model_select, model_score, select_params, **kwargs):
    alipy_al_exps_fcn = functools.partial(alipy_al, qs=qs, model_select=model_select, model_score=model_score, select_params=select_params, **kwargs)
    return alipy_al_exps_fcn

def alipy_al(round, train_id, test_id, Lcollection, Ucollection, saver, examples, labels, global_parameters, qs, model_select, model_score, select_params, **kwargs):
    configs = kwargs['configs']
    seed = kwargs['seed']
    # file = open(f'{configs.data_set}-{configs.qs_name}-{configs.hs_name}-{configs.gs_name}-{configs.exp_name}-detail.csv', 'a')
    # quota
    quota = kwargs['configs'].total_budget
    if quota is None:
        quota = len(Ucollection)
    al_round = len(Lcollection.index)  # num init labelled pool

    # Initialize the model by D_l
    start_train_time = time.time()
    model_select.fit(X=examples[Lcollection.index, :],
                     y=labels[Lcollection.index])
    exec_train_time = time.time() - start_train_time

    # evaluation by score model
    model_score.fit(X=examples[Lcollection.index, :],
                    y=labels[Lcollection.index])
    E_trn_score_curr = model_score.score(examples[train_id, :], labels[train_id])
    E_tst_score_curr = model_score.score(examples[test_id, :], labels[test_id])
    y_pred = model_score.predict(examples[test_id, :])
    confusion_mat_curr = confusion_matrix(labels[test_id], y_pred).ravel()
    # file.write(f'{seed}|{al_round}|{E_tst_score_curr}|{exec_train_time}|\n')
    logging_print('init', f'|{seed}|{al_round}|{E_tst_score_curr}|{exec_train_time:.3f}|')

    # save intermediate results
    st = State(select_index=None,
               performance=[E_tst_score_curr, confusion_mat_curr, E_trn_score_curr])
    saver.add_state(st)

    while quota > 0:
        # sync model into select params for some qs needs model as the param
        if "model" in select_params:
            select_params["model"] = model_select

        start_query_time = time.time()
        select_index = qs.select(Lcollection, Ucollection, **select_params)
        exec_query_time = time.time() - start_query_time

        # update D_l, D_u
        Ucollection.difference_update(select_index)
        Lcollection.update(select_index)

        al_round = len(Lcollection.index)

        # update model, fit current selection
        start_train_time = time.time()
        model_select.fit(X=examples[Lcollection.index, :], y=labels[Lcollection.index])
        exec_train_time = time.time() - start_train_time

        # evaluation by score model
        model_score.fit(X=examples[Lcollection.index, :], y=labels[Lcollection.index])
        E_lbl_score_curr = model_score.score(examples[Lcollection.index, :], labels[Lcollection.index])
        E_trn_score_curr = model_score.score(examples[train_id, :], labels[train_id])
        E_tst_score_curr = model_score.score(examples[test_id, :], labels[test_id])
        y_pred = model_score.predict(examples[test_id, :])
        confusion_mat_curr = confusion_matrix(labels[test_id], y_pred).ravel()
        # file.write(f'{seed}|{al_round}|{E_tst_score_curr}|{exec_train_time}|{exec_query_time}\n')
        logging_print('update', f'|{seed}|{al_round}|{E_tst_score_curr:.3f}|{exec_train_time:.3f}|{exec_query_time}')

        # save intermediate results
        st = State(select_index=select_index,
                   performance=[E_lbl_score_curr, E_trn_score_curr, E_tst_score_curr, confusion_mat_curr, al_round])
        saver.add_state(st)

        # update round
        quota -= 1  # TODO query batch = 1

    # file.close()
    return st


def alipy_al_getres(res):
    # results
    hist_info = {
        "E_lbl_score": [], "E_trn_score": [], "E_tst_score": [], 'confusion_mat': [], 'al_round':[],
    }

    res_curr = res.get_state(0)  # init model score
    hist_info['E_ini_trn_score'] = res_curr['performance'][2]
    hist_info['E_ini_score'] = res_curr['performance'][0]
    hist_info['confusion_mat_ini'] = res_curr['performance'][1]

    for i in range(1, len(res)):
        res_curr = res.get_state(i)
        # evaluation by select model and score model
        # In ALiPy, it has been stored in `acethread.get_results()`
        hist_info["E_lbl_score"].append(res_curr["performance"][0])
        hist_info["E_trn_score"].append(res_curr["performance"][1])
        hist_info["E_tst_score"].append(res_curr["performance"][2])
        hist_info["confusion_mat"].append(res_curr["performance"][3])
        hist_info["al_round"].append(res_curr["performance"][4])
    return hist_info
