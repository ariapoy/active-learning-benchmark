import pandas as pd
HGs = ("LR", "RBFSVM", "RandomForest")
for query in HGs:
    for task in HGs:
        res = pd.read_csv(f"ex8a-margin-zhan-{query}-{task}-RS_noFix_scale-aubc.csv")
        res_aubcs = res['res_tst_score']
        print(f'QueryxTask: {query}x{task} | {res_aubcs.mean():.4f}')
