import pandas as pd
query_models = ['LR', 'RBFSVM', 'RandomForest']
task_models = ['LR', 'RBFSVM', 'RandomForest']
datasets = ['sonar', 'ionosphere', 'gcloudb', 'checkerboard', 'banana', 'haberman', 'tic', 'appendicitis', 'breast', 'twonorm']
aubc_avg_3x3 = {}
aubc_std_3x3 = {}
for data in datasets:
    aubc_avg_3x3[data] = pd.DataFrame(index=query_models, columns=task_models)
    aubc_std_3x3[data] = pd.DataFrame(index=query_models, columns=task_models)
    for q_model in query_models:
        for t_model in task_models:
            if q_model == 'RBFSVM' and t_model == 'RBFSVM':
                aubc = pd.read_csv(f'{data}-margin-zhan-google-zhan-zhan-RS_noFix_scale-aubc.csv')
            else:
                aubc = pd.read_csv(f'{data}-margin-zhan-{q_model}-{t_model}-RS_noFix_scale-aubc.csv')

            aubc_avg = aubc['res_tst_score'].mean()
            aubc_std = aubc['res_tst_score'].std()
            aubc_avg_3x3[data].loc[q_model, t_model] = aubc_avg
            aubc_std_3x3[data].loc[q_model, t_model] = aubc_std

for data in aubc_avg_3x3:
    print(data)
    print(aubc_avg_3x3[data])
    bst_q_model = aubc_avg_3x3[data].astype(float).max(axis=1).idxmax()
    bst_t_model = aubc_avg_3x3[data].astype(float).max().idxmax()
    assert aubc_avg_3x3[data].loc[bst_q_model, bst_t_model] == aubc_avg_3x3[data].astype(float).max().max()
    print(f'Query model: {bst_q_model: <12} X Task model: {bst_t_model}')
    print()

