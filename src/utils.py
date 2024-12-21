import threading
import logging
import os
import sys
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

import time

logger = logging
seed = {}

# initial datasets
def init_data_exps(X, y, seed, init_lbl_size, tst_ratio, init_trn_tst='RS', init_trn_tst_fixSeed='noFix', init_lbl_ubl='RS'):
    '''
    init_trn_tst: 'RS', 'SameDist'
    init_trn_tst_fixSeed: 'noFix', 'Fix'
    init_lbl_ubl: 'RS', 'SameDist', 'nShot'
    '''
    # training and testing sets
    idx = np.arange(X.shape[0])
    trn_size = int(idx.shape[0]*(1 - tst_ratio))
    tst_size = idx.shape[0] - trn_size

    if init_trn_tst_fixSeed:
        rng_trntst = np.random.default_rng(0)
        rng_trntst.shuffle(idx)
    else:
        rng_trntst = np.random.default_rng(seed)
        rng_trntst.shuffle(idx)

    if init_trn_tst == 'RS':  # random splitting of training and testing sets
        idx_trn = idx[:trn_size]
        idx_tst = idx[trn_size:]
    elif init_trn_tst == 'SameDist':
        # TODO maybe update idx after shuffle
        idx_n, idx_p = idx[y[idx]==-1], idx[y[idx]==1]
        ratio_n = idx_n.shape[0]/(idx_n.shape[0] + idx_p.shape[0])
        ratio_p = 1 - ratio_n
        size_n_tst, size_p_tst = int(round(tst_size*ratio_n)), int(round(tst_size*ratio_p))
        rng_trntst.shuffle(idx_n)
        rng_trntst.shuffle(idx_p)
        idx_tst = np.append(idx_n[:size_n_tst], idx_p[:size_p_tst])
        idx_trn = np.setdiff1d(idx, idx_tst)
        rng_trntst.shuffle(idx_trn)

    rng_trntst = np.random.default_rng(seed)
    # Get X_trn, X_tst, X_lbl, X_ubl ; y_trn, y_tst, y_lbl, y_ubl
    X_trn, y_trn = X[idx_trn, :], y[idx_trn]
    # labelled and unlabelled pools
    if 'nShot' == init_lbl_ubl:
        class_dict = {c:idx_trn[y_trn==c] for c in np.unique(y_trn)}
        for c in class_dict:
            rng_trntst.shuffle(class_dict[c])

        idx_lbl = []
        n = 0
        while len(idx_lbl) <= init_lbl_size:
            for c in class_dict:
                idx_lbl.append(class_dict[c][n])
            n += 1

        idx_lbl = np.array(idx_lbl)
        idx_lbl = idx_lbl[:init_lbl_size]
        idx_ubl = np.setdiff1d(idx_trn, idx_lbl)
        rng_trntst.shuffle(idx_ubl)
        idx_trn = np.append(idx_lbl, idx_ubl)
    elif 'SameDist' == init_lbl_ubl:
        # TODO. SameDist can be the function
        idx_n, idx_p = idx_trn[y_trn==-1], idx_trn[y_trn==1]
        ratio_n = idx_n.shape[0]/(idx_n.shape[0] + idx_p.shape[0])
        ratio_p = 1 - ratio_n
        size_n, size_p = int(round(init_lbl_size*ratio_n)), int(round(init_lbl_size*ratio_p))
        rng_trntst.shuffle(idx_n)
        rng_trntst.shuffle(idx_p)
        idx_lbl = np.append(idx_n[:size_n], idx_p[:size_p])
        rng_trntst.shuffle(idx_lbl)
        idx_ubl = np.setdiff1d(idx_trn, idx_lbl)
        idx_trn = np.append(idx_lbl, idx_ubl)
    else:  # if init_lbl_ubl == 'RS':
        rng_trntst.shuffle(idx_trn)
        idx_lbl = idx_trn[:init_lbl_size]
        idx_ubl = idx_trn[init_lbl_size:]

    assert (idx_lbl == idx_trn[:idx_lbl.shape[0]]).all(), 'inconsistent of idx_lbl and idx_trn'
    return idx, idx_trn, idx_tst, idx_lbl, idx_ubl

def logging_setup(export_path):
    logger.basicConfig(
        level=logger.DEBUG,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M:%S',
        filename='{0}.csv'.format(export_path),
        filemode='a')
    # console = logger.StreamHandler()
    console = logging.StreamHandler(stream=None)
    console.setLevel(logger.INFO)
    formatter = logger.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logger.getLogger().addHandler(console)

def logging_set_seed(_seed):
    tid = threading.get_ident()
    seed[tid] = _seed

def logging_print(topic, message, is_print=True, level='info'):
    tid = threading.get_ident()
    if is_print:
        if level == 'info':
                # logger.info('[{}] seed {}: {}'.format(topic, seed[tid], message))
                # tool="alipy" => seed.get(tid) could return None
                # It seems alipy.aceThreading creates new the thread every time.
                logger.info('[{}] seed {}: {}'.format(topic, seed.get(tid), message))
        elif level == 'error':
            # logger.info('[{}] seed {}: {}'.format(topic, seed[tid], message))
            logger.info('[{}] seed {}: {}'.format(topic, seed.get(tid), message))
        else:
            raise NotImplementedError(f'level = {level}')

        for h in logger.getLogger().handlers:
            h.flush()

def logging_update_detail(export_name, update_tst_acc):
    # update detail
    if os.path.isfile(f'{export_name}-detail.csv'):
        logfile = pd.read_csv(f'{export_name}-detail.csv', sep='|', header=None)
        logfile.columns = ['msg', 'seed', 'round', 'tst_acc', 'trn_time', 'qry_time', 'qry_idx']
        logfile = logfile.dropna(subset=['seed', 'round'])
        logfile['seed'] = logfile['seed'].astype(int)
        logfile['round'] = logfile['round'].astype(int)
        logfile = logfile.set_index(['seed', 'round'])
        for key in update_tst_acc:
            # logfile['tst_acc'] = logfile['tst_acc'].map(update_tst_acc)
            logfile.loc[key, 'tst_acc'] = update_tst_acc[key]

        logfile = logfile[~logfile.index.duplicated(keep='last')]
        logfile = logfile.sort_index()
        logfile = logfile.reset_index()
        logfile = logfile[['msg', 'seed', 'round', 'tst_acc', 'trn_time', 'qry_time', 'qry_idx']]
        logfile['msg'] = logfile['msg'].str[:11].drop_duplicates(keep='first')  # keep datetime
        logfile.to_csv(f'{export_name}-detail.csv', sep='|', index=None, header=False)

def AUBC(quota, resseq, bsize=1):
    ressum = 0.0
    quota = len(quota)
    if quota % bsize == 0:
        for i in range(len(resseq)-1):
            ressum = ressum + (resseq[i+1] + resseq[i]) * bsize / 2
    else:
        for i in range(len(resseq)-2):
            ressum = ressum + (resseq[i+1] + resseq[i]) * bsize / 2
        k = quota % bsize
        ressum = ressum + ((resseq[-1] + resseq[-2]) * k / 2)
    ressum = round(ressum / quota, 5)
    return ressum
