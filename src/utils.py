import threading
import logging
import os
import sys
from sklearn.metrics import confusion_matrix

import time

logger = logging
seed = {}

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

def AUBC_Zhan(quota, resseq, bsize=1):
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