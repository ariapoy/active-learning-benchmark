import traceback
import argparse
from headers import *
from utils import *
from config import SelectModelBuilder, ScoreModelBuilder, QueryStrategyBuilder

from algo.bso import BSO, bso_al
from algo.alipy import alipy_al_exps, alipy_al, alipy_al_getres
from algo.google import select_batch, google_al
from algo.libact import libact_al, AUBC_Zhan

def exp_compute(seed, data_set, qs_name, hs_name, tst_size, init_lbl_size, module="google", **kwargs):
    data = load_svmlight_file(
        "../data/dataset_used_in_ALSurvey/{0}-svmstyle.txt".format(data_set))
    X, y = data[0], data[1]
    X = np.asarray(X.todense())
    if np.unique(y).shape[0] == 2:  # binary
        if -1 not in set(y):
            y[y==0] = -1  # bug for hintsvm, dwus?
    else:
        pass

    # initial setttings
    # total budget
    quota = None
    # training and testing sets
    idx = np.arange(X.shape[0])
    trn_size = int(idx.shape[0]*(1 - tst_size))
    tst_size = idx.shape[0] - trn_size
    if '_' in args.exp_name:
        init_trn_tst = args.exp_name.split('_')[0]  # '{RS, SameDist}_{fix, noFix}'
        init_trn_tst_fixSeed = args.exp_name.split('_')[1]
    else:
        init_trn_tst = 'SameDist'
        init_trn_tst_fixSeed = True

    init_trn_tst = init_trn_tst if init_trn_tst in {'RS', 'SameDist'} else 'SameDist'
    init_trn_tst_fixSeed = not ('noFix' == init_trn_tst_fixSeed)  # True

    if init_trn_tst_fixSeed:
        rng_trntst = np.random.default_rng(0)
        rng_trntst.shuffle(idx)
    else:
        # print('D_tst is not fix')
        rng_trntst = np.random.default_rng(seed)
        rng_trntst.shuffle(idx)

    if init_trn_tst == 'RS':  # random splitting of training and testing sets
        # print('D_trn is RS')
        # fix "index" of datasets.
        # shuffle the index, split by this new order
        # get size of train, if tst_size in [0, 1]
        # get the training and testing
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
    init_lbl_ubl = args.exp_name  # 'RS', 'SameDist', 'nShot'
    if 'nShot' in init_lbl_ubl:
        idx_n, idx_p = idx_trn[y_trn==-1], idx_trn[y_trn==1]
        ratio_n = 1/2
        ratio_p = 1 - ratio_n
        size_n, size_p = int(round(init_lbl_size*ratio_n)), int(round(init_lbl_size*ratio_p))
        if size_n != size_p:
            size_n = min(size_n, size_p)
            size_p = size_n
        rng_trntst.shuffle(idx_n)
        rng_trntst.shuffle(idx_p)
        idx_lbl = np.append(idx_n[:size_n], idx_p[:size_p])
        rng_trntst.shuffle(idx_lbl)
        idx_ubl = np.setdiff1d(idx_trn, idx_lbl)
        idx_trn = np.append(idx_lbl, idx_ubl)
    elif 'SameDist' in init_lbl_ubl:
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
        # print('D_l is RS')
        rng_trntst.shuffle(idx_trn)
        idx_lbl = idx_trn[:init_lbl_size]
        idx_ubl = idx_trn[init_lbl_size:]

    assert (idx_lbl == idx_trn[:idx_lbl.shape[0]]).all(), 'inconsistent of idx_lbl and idx_trn'
    # Get X_trn, X_tst, X_lbl, X_ubl ; y_trn, y_tst, y_lbl, y_ubl
    X_trn, y_trn = X[idx_trn, :], y[idx_trn]
    X_tst, y_tst = X[idx_tst, :], y[idx_tst]
    X_lbl, y_lbl = X[idx_lbl, :], y[idx_lbl]

    # Quota (Budget)
    quota = ubl_len = idx_ubl.shape[0]

    # Make sure each class with at least one sample
    try:
        y_class, y_cnt = np.unique(y, return_counts=True)
        y_trn_class, y_trn_cnt = np.unique(y_trn, return_counts=True)
        y_lbl_class, y_lbl_cnt = np.unique(y_lbl, return_counts=True)
        assert len(y_class) == len(y_trn_class) == len(y_lbl_class)
        assert (y_trn_cnt >= 1).all()
        assert (y_lbl_cnt >= 1).all()
    except:
        if len(y_trn_class) != len(y_class):
            results = "Not enough label in training set"
        elif len(y_lbl_class) != len(y_class):
            results = "Not enough label in label pool"
        elif (y_trn_cnt < 1).any():
            idx_class = np.where(y_trn_cnt < 1)
            res_class = y_trn_class[idx_class]
            results = "Not enough label of class {0} in traininig set".format(
                res_class)
        elif (y_lbl_cnt < 1).any():
            idx_class = np.where(y_lbl_cnt < 1)
            res_class = y_lbl_class[idx_class]
            results = "Not enough label of class {0} in label pool".format(
                res_class)

        logging_print('data', f'|{results}|||||', level='error')
        return seed, results, range(quota)

    # Add Zhan's data preprocessing (personal message)
    if "scale" in args.exp_name:
        # print('scale')
        scaler = StandardScaler()
        X_trn = scaler.fit_transform(X_trn)
        X = scaler.transform(X)
        X_tst = X[idx_tst, :]
        X_lbl = X[idx_lbl, :]

    # Selecting model
    model_select = SelectModelBuilder(hs_name)
    # Scoring model
    if kwargs['gs_name']:
        model_score = ScoreModelBuilder(gs_name)
    else:
        model_score = copy.deepcopy(model_select)

    # Query strategy
    qs_dict = QueryStrategyBuilder(qs_name)

    # Labeler
    lbr = None

    # Load libact query strategies
    if module == "libact":
        ubl_len = idx_ubl.shape[0]

        # libact.Dataset
        trn_ds = libact_Dataset(
            X_trn, np.concatenate([y_lbl, [None] * ubl_len]))
        tst_ds = libact_Dataset(X_tst, y_tst)

        # libact.models
        if isinstance(model_select, list):
            model_select_libact = [copy.deepcopy(m) for m in model_select]
        elif model_select is None:
            model_select_libact = [copy.deepcopy(model_score)]
        else:
            model_select_libact = [copy.deepcopy(model_select)]

        model_select_libact_new = []
        for m in model_select_libact:
            if getattr(m, 'predict_proba', 'not_prob_out') == 'not_prob_out':
                model_select_libact_new.append(libact_skContiAdapter(m))
            else:  # if m.predict_proba exist
                model_select_libact_new.append(libact_skProbaAdapter(m))
        model_select_libact = model_select_libact_new

        # libact.qs
        if "model" in qs_dict["params"]:
            qs_dict["params"]["model"] = model_select_libact[0]
        elif "models" in qs_dict["params"]:
            qs_dict["params"]["models"] = model_select_libact

        if "T" in qs_dict["params"]:
            qs_dict["params"]["T"] = quota

        if "query_strategies" in qs_dict["params"]:
            qs_dict["params"]["query_strategies"] = [sub_qs(trn_ds) for sub_qs in qs_dict['params']['query_strategies']]

        qs = qs_dict["qs"](trn_ds, **qs_dict["params"])

        # hintsvm: align query model and select model
        if qs_name == 'hintsvm':
            queryModel_params = model_select_libact[0]._model.get_params()
            n_features = trn_ds._X.shape[1]
            X_var = trn_ds._X.var()
            for k in qs.svm_params:
                if k in queryModel_params:
                    if queryModel_params[k] == 'auto':
                        queryModel_params[k] = 1 / n_features
                    elif queryModel_params[k] == 'scale':
                        queryModel_params[k] = 1 / (n_features * X_var)

                    qs.svm_params[k] = queryModel_params[k]

        # libact.labeler
        fully_labeled_trn_ds = libact_Dataset(X_trn, y_trn)
        lbr = IdealLabeler(fully_labeled_trn_ds)

        # Run active learning algorithm
        try:
            results = libact_al(trn_ds, tst_ds, fully_labeled_trn_ds,
                                qs, model_select_libact, model_score, quota, lbr, seed=seed, configs=args,
                                idxs=[idx, idx_trn, idx_tst, idx_lbl])
        except Exception as e:
            results = traceback.format_exc()
            logging_print('framework libact', f'|Error by {results}|||||', level='error')

    elif module == "google":
        ubl_len = idx_ubl.shape[0]

        # google.Dataset
        # pass

        # google.models
        if model_select is None:
            model_select = copy.deepcopy(model_score)

        # google.qs
        uniform_qs = AL_MAPPING["uniform"](X_trn, y_trn, seed)
        qs = qs_dict["qs"](X_trn, y_trn, seed)

        # google.labeler
        # pass

        # Run active learning algorithm
        try:
            results = google_al(X_trn, y_trn, X_tst, y_tst, idx_lbl,
                                qs, uniform_qs, model_select, model_score, quota, batch_size=1,
                                X_all=X, y_all=y, indices=idx, seed=seed, configs=args,
                                idxs=[idx, idx_trn, idx_tst, idx_lbl])  # check
        except Exception as e:
            logging_print('framework', f'|Error by {e}|||||', level='error')
            results = e

    elif module == "alipy":
        # alipy.Dataset
        alipy_trn, alipy_tst, alipy_lbl, alipy_ubl = [idx_trn], [idx_tst], [idx_lbl], [idx_ubl]

        y = y.astype(int)
        label_class = np.unique(y).tolist()
        label_class = sorted(label_class)
        if label_class != [0, 1]:
            label_map = {k: v for k, v in zip(
                label_class, range(len(label_class)))}
            y = pd.Series(y)
            y = y.map(label_map)
            y = y.values

        # alipy.models
        if model_select is None:
            model_select = copy.deepcopy(model_score)

        # alipy.qs
        qs = qs_dict["qs"](X, y, **qs_dict["params"])
        qs_select_params = qs_dict["select"]

        # alipy.labeler
        # pass

        try:
            acethread = aceThreading(examples=X, labels=y,
                                     train_idx=alipy_trn, test_idx=alipy_tst,
                                     label_index=alipy_lbl, unlabel_index=alipy_ubl,
                                     max_thread=None, refresh_interval=1*60*60,
                                     saving_path='./alipy-log/')
            acethread.set_target_function(alipy_al_exps(
                qs, model_select, model_score, select_params=qs_select_params, seed=seed, configs=args))
            acethread.start_all_threads(global_parameters=None)
            # get results of exps
            stateIO_list = acethread.get_results()
            res = stateIO_list[0]
            results = alipy_al_getres(res)
            # save the state of multi_thread to the saving_path in pkl form
            acethread.save()
            del acethread
        except Exception as e:
            results = e
            logging_print('framework', f'|Error by {e}|||||', level='error')

    elif module == "bso":
        if model_select is None:
            model_select = copy.deepcopy(model_score)

        num_beam = qs_dict["params"]["num_beam"]
        results = bso_al(X, y, idx_trn, idx_tst, idx_lbl,model_select,
                         model_score, num_beam=num_beam, method=kwargs.get('lookDtst'),
                         seed=seed, configs=args)

    return seed, results, range(quota)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Survey experiments on active learning.')
    # query strategy
    parser.add_argument('--qs_name', dest="qs_name",
                        help='Name of query strategy',
                        default="us-zhan", type=str)
    # query strategy--bso
    parser.add_argument('--lookDtst', dest='lookDtst',
                        help='type of bso',
                        default='lookDtst', type=str)
    # hypothesis set/ models
    parser.add_argument('--hs_name', dest="hs_name",
                        help='Name of query model/hypothesis set',
                        default="us-zhan", type=str)
    parser.add_argument('--gs_name', dest="gs_name",
                        help='Name of task model/hypothesis set',
                        default="zhan", type=str)
    # exps
    parser.add_argument('--seed', dest='seed',
                        help='Random state seed for reproducing',
                        default=0, type=int)
    parser.add_argument('--n_jobs', dest='n_jobs',
                        help='Multiprocessing of exps with seed',
                        default=1, type=int)
    parser.add_argument('--n_trials', dest='n_trials',
                        help='Number of trials, starting from seed',
                        default=1, type=int)
    # dataset
    parser.add_argument('--data_set', dest='data_set',
                        help='provide the file name file for running',
                        default="heart", type=str)
    parser.add_argument('--tst_size', dest='tst_size',
                        help='Size of testing set. (0, 1] fraction of whole data',
                        default=0.4, type=float)
    parser.add_argument('--init_lbl_size', dest='init_lbl_size',
                        help='Size of initial label pool. INT',
                        default=20, type=int)
    # env
    parser.add_argument('--exp_name', dest='exp_name',
                        help='exp_name name',
                        default="RS_noFix_scale", type=str)
    # module
    parser.add_argument('--tool', dest='tool',
                        help='Package name',
                        default="libact", type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if 'bso' in args.qs_name:
        args.exp_name += f'_{args.lookDtst}'

    # module
    tool_name = args.tool
    # query strategy
    qs_name, hs_name, gs_name = args.qs_name, args.hs_name, args.gs_name
    if gs_name == 'None':
        gs_name = None
    # exps
    seed, n_jobs, n_trials = args.seed, args.n_jobs, args.n_trials
    # dataset configs
    data_set, tst_size, init_lbl_size = args.data_set, args.tst_size, args.init_lbl_size
    # env
    exp_name = args.exp_name
    # path
    export_name =  f'{data_set}-{qs_name}-{hs_name}-{gs_name}-{exp_name}'
    # logger
    logging_setup(f'{export_name}-detail')
    # log
    logging_print('exp', f'|{export_name} Start|||||', level='info')

    # for multi processing
    other_configs = {'lookDtst': args.lookDtst, 'args': args, 'gs_name': gs_name,}
    def run(seed):
        logging_setup(f'{export_name}-detail')
        logging_set_seed(seed);
        results = exp_compute(seed, data_set, qs_name, hs_name,
                              tst_size, init_lbl_size, module=tool_name, **other_configs)
        return results

    # main/core executation
    expno_range = range(seed, seed + n_trials)

    start_time = time.time()
    # alipy needs multiprocessing
    if tool_name == "alipy" or "quire" in qs_name:
        res_list = []
        if n_jobs == 1:
            for seed in tqdm(expno_range):
                res_list.append(run(seed))
        else:
            with Pool(n_jobs) as p:
                imap_unordered_it = p.imap_unordered(run, expno_range)
                for res in imap_unordered_it:
                    res_list.append(res)
    else:
        # reference: https://joblib.readthedocs.io/en/latest/parallel.html#avoiding-over-subscription-of-cpu-ressources
        with parallel_backend('loky', inner_max_num_threads=5):
            res_list = Parallel(n_jobs=n_jobs)(delayed(run)(seed)
                                               for seed in tqdm(expno_range))

    res_dict = {
        "res_expno": [],
        "res_lbl_score": [],
        "res_tst_score": []
    }
    error_log = {}
    learn_curve = {}
    confusion_mats = {}
    update_tst_acc = {}
    for expno, res, budget in res_list:
        if isinstance(res, dict):
            E_lbl_score_curr = AUBC_Zhan(budget, res["E_lbl_score"])
            E_tst_score_curr = AUBC_Zhan(budget, res["E_tst_score"])
            res_dict["res_expno"].append(expno)
            res_dict["res_lbl_score"].append(E_lbl_score_curr)
            res_dict["res_tst_score"].append(E_tst_score_curr)
            learn_curve[expno] = [res['E_ini_score']] + res['E_tst_score']
            confusion_mats[expno] = np.vstack([res['confusion_mat_ini']] + res['confusion_mat'])
            # update detail
            for rnd, tst in enumerate(res["E_tst_score"]):
                rnd = rnd + init_lbl_size + 1
                update_tst_acc[(expno, rnd)] = tst
        else:
            if error_log.get(str(res)):
                error_log[str(res)] += ",{0}".format(expno)
            else:
                error_log[str(res)] = "{0}".format(expno)

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

    del res_list
    res = pd.DataFrame(res_dict)
    learn_curve = pd.DataFrame(learn_curve)
    learn_curve.index = [init_lbl_size] + [init_lbl_size+1+b for b in budget]
    learn_curve = learn_curve.T
    # TODO only support binary classification
    # confusion_mats = {k: pd.DataFrame(confusion_mats[k]) for k in confusion_mats}
    # confusion_mats_fin = []
    # for k in confusion_mats:
    #     confusion_mats[k]['expno'] = k
    #     confusion_mats[k]['round'] = [init_lbl_size] + [init_lbl_size+1+b for b in budget]
    #     confusion_mats_fin.append(confusion_mats[k])
    # confusion_mats_fin = pd.concat(confusion_mats_fin, ignore_index=True)
    # confusion_mats_fin.columns = ['tn', 'fp', 'fn', 'tp', 'expno', 'round']
    # confusion_mats_fin = confusion_mats_fin[['expno', 'round', 'tn', 'fp', 'fn', 'tp']]
    # confusion_mats_fin = confusion_mats_fin.set_index('expno')

    res["res_lbl_score"].mean(), res["res_tst_score"].mean()

    if os.path.isfile(f'{export_name}-aubc.csv'):
        res.to_csv(f'{export_name}-aubc.csv', index=None, mode="a", header=None)
        # drop duplicates
        res = pd.read_csv(f'{export_name}-aubc.csv')
        res.columns = ['res_expno', 'res_lbl_score', 'res_tst_score']
        res = res.drop_duplicates(keep='last')
        res.to_csv(f'{export_name}-aubc.csv', index=None)

        # export confusion matrix
        # confusion_mats_fin.to_csv(f'{export_name}-CM.csv', mode="a", header=None)
        # res = pd.read_csv(f'{export_name}-CM.csv')
        # res.columns = ['expno', 'round', 'tn', 'fp', 'fn', 'tp']
        # res = res.drop_duplicates(keep='last')
        # res.to_csv(f'{export_name}-CM.csv', index=None)
    else:
        res.to_csv(f'{export_name}-aubc.csv', index=None)
        # confusion_mats_fin.to_csv(f'{export_name}-CM.csv')

    if len(error_log) > 0:
        logging_print('algo', f'|{repr(error_log)}|||||', level='error')

    logging_print('exp', f'|{export_name} End|||||', level='info')
