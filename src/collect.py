import pandas as pd
import os

import pdb

# collect the same prefix within the method (directory)
files = os.listdir()
datanames = []
files = [f for f in files if ('.csv' in f) and ('final' not in f) and ('-lc' not in f)]
for f in files:
    if ("google" in f) or ("libact" in f) or ("alipy" in f) or ('bso' in f):
        datanames.append(f.split(".")[0].split("-")[3])
    else:
        datanames.append(f.split(".")[0].split("-")[2])

files = sorted(zip(files, datanames), key=lambda x: x[1])

staged_files = []
cur_file = ""
for f, dataname in files:
    if dataname == cur_file:
        staged_files.append(f)
    elif cur_file == "":  # for first term
        cur_file = dataname
        staged_files.append(f)
    else:
        # merge the data source with the same dataname
        data = [pd.read_csv(f) for f in staged_files]
        data = pd.concat(data)
        # drop duplicates of the same expno
        data = data.drop_duplicates(subset=['res_expno'], keep="first")
        # export to final res
        data.to_csv("{0}-final.csv".format(cur_file), index=None)
        # reset tmp
        cur_file = dataname
        staged_files = [f]

# for last term
# merge the data source with the same dataname
data = [pd.read_csv(f) for f in staged_files]
data = pd.concat(data)
# drop duplicates of the same expno
data = data.drop_duplicates(subset=['res_expno'], keep="first")
# export to final res
data.to_csv("{0}-final.csv".format(cur_file), index=None)

