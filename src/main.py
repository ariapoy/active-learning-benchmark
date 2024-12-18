import traceback
import argparse
from headers import *
from utils import *
from sklearn.preprocessing import LabelEncoder
from config import SelectModelBuilder, ScoreModelBuilder, QueryStrategyBuilder

from algo.bso import BSO, bso_al
from algo.alipy import alipy_al_exps, alipy_al, alipy_al_getres
from algo.google import select_batch, google_al
from algo.libact import libact_al, AUBC
from algo.skactiveml import skactiveml_al

def exp_compute(seed, data_set, qs_name, hs_name, tst_ratio, init_lbl_size, module="google", **kwargs):
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
        if hs_name == 'XGBoost':
            # mapping y to [0, 1, 2, ...]
            y = LabelEncoder().fit_transform(y)

    # initial setttings
    # training and testing sets
    idx, idx_trn, idx_tst, idx_lbl, idx_ubl = init_data_exps(X, y, seed, init_lbl_size, tst_ratio, init_trn_tst='RS', init_trn_tst_fixSeed='noFix', init_lbl_ubl='RS')
    # Get X_trn, X_tst, X_lbl, X_ubl ; y_trn, y_tst, y_lbl, y_ubl
    X_trn, y_trn = X[idx_trn, :], y[idx_trn]
    X_tst, y_tst = X[idx_tst, :], y[idx_tst]
    X_lbl, y_lbl = X[idx_lbl, :], y[idx_lbl]

    # Quota (Budget)
    if args.total_budget and idx_ubl.shape[0]>args.total_budget:
        quota = ubl_len = args.total_budget
    else:
        print('Use the size of unlabeled as the total budget.')
        quota = ubl_len = idx_ubl.shape[0]
        # TODO. put all arguments to args
        args.total_budget = quota

    # Make sure each class with at least one sample
    y_class, y_cnt = np.unique(y, return_counts=True)
    y_trn_class, y_trn_cnt = np.unique(y_trn, return_counts=True)
    y_lbl_class, y_lbl_cnt = np.unique(y_lbl, return_counts=True)
    if len(y_trn_class) != len(y_class):
        results = "Not enough label in training set"
        return seed, results, range(quota)
    elif len(y_lbl_class) != len(y_class):
        results = "Not enough label in label pool"
        return seed, results, range(quota)
    elif (y_trn_cnt < 1).any():
        idx_class = np.where(y_trn_cnt < 1)
        res_class = y_trn_class[idx_class]
        results = "Not enough label of class {0} in traininig set".format(
            res_class)
        return seed, results, range(quota)
    elif (y_lbl_cnt < 1).any():
        idx_class = np.where(y_lbl_cnt < 1)
        res_class = y_lbl_class[idx_class]
        results = "Not enough label of class {0} in label pool".format(
            res_class)
        return seed, results, range(quota)

    if args.scale:
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
        results = libact_al(trn_ds, tst_ds, fully_labeled_trn_ds,
                            qs, model_select_libact, model_score, quota, lbr, seed=seed, configs=args,
                            idxs=[idx, idx_trn, idx_tst, idx_lbl])
        # except Exception as e:
        #     results = traceback.format_exc()
        #     logging_print('framework libact', f'|Error by {results}|||||', level='error')

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
        results = google_al(X_trn, y_trn, X_tst, y_tst, idx_lbl,
                            qs, uniform_qs, model_select, model_score, quota, batch_size=1,
                            X_all=X, y_all=y, indices=idx_trn, seed=seed, configs=args)
        # except Exception as e:
        #     logging_print('framework', f'|Error by {e}|||||', level='error')
        #     results = e

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
        # except Exception as e:
        #     results = e
        #     logging_print('framework', f'|Error by {e}|||||', level='error')

    elif module == "bso":
        if model_select is None:
            model_select = copy.deepcopy(model_score)

        num_beam = qs_dict["params"]["num_beam"]
        results = bso_al(X, y, idx_trn, idx_tst, idx_lbl,model_select,
                         model_score, num_beam=num_beam, method=kwargs.get('lookDtst'),
                         seed=seed, configs=args)

    elif module == "scikital":
        # Dataset
        y_lbl_skal = np.full(shape=y.shape, fill_value=MISSING_LABEL)
        y_lbl_skal[idx_lbl] = y[idx_lbl]
        y_lbl_skal = y_lbl_skal[idx_trn]

        # query-oriented model and task-oriented model
        if model_select is None:
            model_select = copy.deepcopy(model_score)

        model_select_scikital = SklearnClassifier(model_select, classes=np.unique(y_trn))

        # query strategy
        qs = qs_dict["qs"](**qs_dict["params"])

        # labeler
        # y_trn

        # Run active learning algorithm
        results = skactiveml_al(X_trn, y_lbl_skal, X_tst, y_tst, y_trn, qs, model_select_scikital, model_score, quota, seed=seed, configs=args,
                                y_all=y, idx_trn=idx_trn, qs_name=qs_name)

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
                        default="XGBoost", type=str)
    parser.add_argument('--gs_name', dest="gs_name",
                        help='Name of task model/hypothesis set',
                        default="XGBoost", type=str)
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

    parser.add_argument('--total_budget', dest="total_budget",
                        help='Budget of quota',
                        default=None, type=int)
    parser.add_argument('--scale', action='store_true',
                        help='Scale the data or not, we use StandardScaler')

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
    if n_jobs == 1:
        res_list = []
        for seed in tqdm(expno_range):
            res_list.append(run(seed))

    else:
        if tool_name == "alipy" or "quire" in qs_name:
            res_list = []
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
    update_tst_acc = {}
    for expno, res, budget in res_list:
        if isinstance(res, dict):
            E_lbl_score_curr = AUBC(budget, res["E_lbl_score"])
            E_tst_score_curr = AUBC(budget, res["E_tst_score"])
            res_dict["res_expno"].append(expno)
            res_dict["res_lbl_score"].append(E_lbl_score_curr)
            res_dict["res_tst_score"].append(E_tst_score_curr)
            learn_curve[expno] = [res['E_ini_score']] + res['E_tst_score']
            # update detail
            for rnd, tst in enumerate(res["E_tst_score"]):
                rnd = rnd + init_lbl_size + 1
                update_tst_acc[(expno, rnd)] = tst
        else:
            if error_log.get(str(res)):
                error_log[str(res)] += ",{0}".format(expno)
            else:
                error_log[str(res)] = "{0}".format(expno)

    del res_list
    res = pd.DataFrame(res_dict)
    learn_curve = pd.DataFrame(learn_curve)
    learn_curve.index = [init_lbl_size] + [init_lbl_size+1+b for b in budget]
    learn_curve = learn_curve.T

    res["res_lbl_score"].mean(), res["res_tst_score"].mean()

    if os.path.isfile(f'{export_name}-aubc.csv'):
        res.to_csv(f'{export_name}-aubc.csv', index=None, mode="a", header=None)
        # drop duplicates
        res = pd.read_csv(f'{export_name}-aubc.csv')
        res.columns = ['res_expno', 'res_lbl_score', 'res_tst_score']
        res = res.drop_duplicates(keep='last')
        res.to_csv(f'{export_name}-aubc.csv', index=None)
    else:
        res.to_csv(f'{export_name}-aubc.csv', index=None)

    if os.path.isfile(f'{export_name}-learn_curve.csv'):
        learn_curve.to_csv(f'{export_name}-learn_curve.csv', mode="a", header=None)
        # drop duplicates
        learn_curve = pd.read_csv(f'{export_name}-learn_curve.csv', index_col=0,)
        learn_curve = learn_curve.drop_duplicates(keep='last')
        learn_curve.to_csv(f'{export_name}-learn_curve.csv')
    else:
        learn_curve.to_csv(f'{export_name}-learn_curve.csv')

    if len(error_log) > 0:
        logging_print('algo', f'|{repr(error_log)}|||||', level='error')

    logging_print('exp', f'|{export_name} End|||||', level='info')
