from utils import *
import copy
import numpy as np
import pandas as pd


class BSO():
    """BSO Beam Search Orcale Algorithm
    """

    def __init__(self, X, y, idx_trn, idx_tst, idx_lbl, model, num_beam=5, lookDtst=True, logfile=None, seed=None):
        self.X = X
        self.y = y
        self.idx_trn = idx_trn
        self.idx_tst = idx_tst
        self.idx_lbl = idx_lbl
        self.nlbl = self.idx_lbl.shape[0]
        self.model = model
        self.num_beam = num_beam
        self.lookDtst = lookDtst
        self.logfile = logfile
        self.seed = seed

    def _score(self, idx):
        self.model.fit(self.X[idx, :], self.y[idx])
        if self.lookDtst:
            acc = self.model.score(self.X[self.idx_tst, :], self.y[self.idx_tst])
        else:
            acc = self.model.score(self.X[self.idx_trn, :], self.y[self.idx_trn])
        return acc

    def compute(self):
        idx_Dcand_collect, _ = self.BSO_step1(
            self.idx_lbl, num_beam=self.num_beam)
        al_round = len(self.idx_lbl)

        while idx_Dcand_collect[0].shape[0] < self.idx_trn.shape[0]:
            # mqke query and get required sample id
            start_query_time = time.time()
            gap = self.idx_trn.shape[0] - idx_Dcand_collect[0].shape[0]
            if gap < self.num_beam:
                idx_Dcand_collect = self.BSO_step2(
                    idx_Dcand_collect, num_beam=gap)
            else:
                idx_Dcand_collect = self.BSO_step2(
                    idx_Dcand_collect, num_beam=self.num_beam)

            exec_query_time = time.time() - start_query_time
            al_round += 1
            # export log
            # if self.logfile is not None:
            #     self.logfile.write(f'{self.seed}|{al_round}|{None}|{None}|{exec_query_time}\n')
            logging_print('update', f'|{self.seed}|{al_round}|{None}|{None}|{exec_query_time:.3f}')

        idx_opt_seq = idx_Dcand_collect[0]  # unpack to get best sequence
        # export log, we needn't query for last sample
        al_round += 1
        # if self.logfile is not None:
        #     self.logfile.write(f'{self.seed}|{al_round}|{None}|{None}|{None}\n')
        logging_print('final', f'|{self.seed}|{al_round}|{None}|{None}|{None}')

        return idx_opt_seq

    def BSO_step1(self, idx_Dl, num_beam):
        idx_Dli_collect = [np.append(
            idx_Dl, idx_xi) for idx_xi in self.idx_trn[~np.isin(self.idx_trn, idx_Dl)]]
        idx_Dli_collect_score = []
        for Di1_idx in idx_Dli_collect:
            acc_curr = self._score(Di1_idx)
            idx_Dli_collect_score.append(acc_curr)

        # if largest x_{i} more than 5?
        idx_Dli_collect_score_bst_pos = np.argpartition(
            idx_Dli_collect_score, -num_beam)[-num_beam:]
        ask_ids = [idx_Dli_collect[idx_pos]
                   for idx_pos in idx_Dli_collect_score_bst_pos]
        ask_ids_scores = [idx_Dli_collect_score[idx]
                          for idx in idx_Dli_collect_score_bst_pos]
        return ask_ids, ask_ids_scores

    def BSO_step2(self, idx_Dli_collect, num_beam):
        idx_Dlij_collect, idx_Dlij_collect_score = [], []
        for idx_Dli in idx_Dli_collect:
            idx_Dlij, idx_Dlij_score = self.BSO_step1(idx_Dli, num_beam)
            idx_Dlij_collect += idx_Dlij
            idx_Dlij_collect_score += idx_Dlij_score

        # if largest x_{i} more than 5?
        idx_Dlij_collect_score_bst_pos = np.argpartition(
            idx_Dlij_collect_score, -num_beam)[-num_beam:]
        ask_ids = [idx_Dlij_collect[idx_pos]
                   for idx_pos in idx_Dlij_collect_score_bst_pos]
        return ask_ids

# # bso active learning framework


def bso_al(X, y, idx_trn, idx_tst, idx_lbl, model_select, model_score, num_beam=5, method='lookDtst', **kwargs):
    configs = kwargs['configs']
    seed = kwargs['seed']
    # export_name = f'{configs.data_set}-{configs.qs_name}-{configs.hs_name}-{configs.gs_name}-{configs.exp_name}-detail.csv'
    # file = open(export_name, 'a')
    # Results
    hist_info = {
        "E_lbl_score": [], "E_trn_score": [], "E_tst_score": [], 'confusion_mat': []
    }
    n_ubl = len(idx_trn) - len(idx_lbl)
    # quota
    idx_lbl_curr = idx_lbl.copy()
    quota = n_ubl
    al_round = len(idx_lbl_curr)  # num init labelled pool

    # init model
    start_train_time = time.time()
    model_score.fit(X[idx_lbl_curr, :], y[idx_lbl_curr])
    exec_train_time = time.time() - start_train_time

    # evaluation by score model
    E_lbl_score_curr = model_score.score(X[idx_lbl, :], y[idx_lbl])
    E_trn_score_curr = model_score.score(X[idx_trn, :], y[idx_trn])
    E_tst_score_curr = model_score.score(X[idx_tst, :], y[idx_tst])
    y_pred = model_score.predict(X[idx_tst, :])
    confusion_mat_curr = confusion_matrix(y[idx_tst], y_pred).ravel()
    hist_info['E_ini_score'] = E_tst_score_curr
    hist_info['confusion_mat_ini'] = confusion_mat_curr
    # file.write(f'{seed}|{al_round}|{E_tst_score_curr}|{exec_train_time}|\n')
    logging_print('init', f'|{seed}|{al_round}|{E_tst_score_curr:.3f}|{exec_train_time:.3f}|')

    # build BSO and compute opt result
    is_lookDtst = method == 'lookDtst'
    # qs = BSO(X, y, idx_trn, idx_tst, idx_lbl, model_select, num_beam=num_beam, lookDtst=is_lookDtst, logfile=file, seed=seed)
    qs = BSO(X, y, idx_trn, idx_tst, idx_lbl, model_select, num_beam=num_beam, lookDtst=is_lookDtst , seed=seed)
    bst_seq = qs.compute()
    # file.close()

    # evaluate best sequence
    hist_info['exec_train_time'] = []
    for i in range(1, n_ubl + 1):
        # update dataset
        idx_lbl_curr = bst_seq[:len(idx_lbl) + i]
        al_round = len(idx_lbl_curr)

        # evaluation by score model
        # It could change `model.coef_` again
        start_train_time = time.time()
        # update model
        model_score.fit(X[idx_lbl_curr, :], y[idx_lbl_curr])
        exec_train_time = time.time() - start_train_time
        hist_info['exec_train_time'].append(exec_train_time)
        # eval
        # TODO: move evaluation to utils
        E_lbl_score_curr = model_score.score(X[idx_lbl_curr], y[idx_lbl_curr])
        hist_info["E_lbl_score"].append(E_lbl_score_curr)
        E_trn_score_curr = model_score.score(X[idx_trn, :], y[idx_trn])
        hist_info["E_trn_score"].append(E_trn_score_curr)
        E_tst_score_curr = model_score.score(X[idx_tst, :], y[idx_tst])
        hist_info["E_tst_score"].append(E_tst_score_curr)

        # confusion matrix at each round, only for test set
        y_pred = model_score.predict(X[idx_tst, :])
        confusion_mat_curr = confusion_matrix(y[idx_tst], y_pred).ravel()
        hist_info['confusion_mat'].append(confusion_mat_curr)

    return hist_info
