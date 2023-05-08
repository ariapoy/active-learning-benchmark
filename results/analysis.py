#!/usr/bin/env python
# coding: utf-8

# # Re-Benchmark of Pool-Based Active Learning for Binary Classification
# 
# Reproduce all figures and tables in Re-Benchmark of Pool-Based Active Learning for Binary Classification.

# In[1]:


import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats


# In[2]:


qs_list = ['uniform', 'us', 'qbc', 'hintsvm', 'quire', 'albl', 'dwus', 'vr', 'kcenter',  # libact
           'margin', 'graph', 'hier', 'infodiv', 'mcm',  # google
           'eer', 'bmdr', 'spal', 'lal',  # alipy
           'bsoDtst']
al_list = ['us', 'qbc', 'hintsvm', 'quire', 'albl', 'dwus', 'vr', 'kcenter',  # libact
           'margin', 'graph', 'hier', 'infodiv', 'mcm',  # google
           'eer', 'bmdr', 'spal', 'lal',  # alipy
          ]
small_data_list = ["appendicitis", "sonar", "parkinsons", "ex8b", "heart", "haberman", "ionosphere", "clean1",
             "breast", "wdbc", "australian", "diabetes", "mammographic", "ex8a", "tic", "german",
             "splice", "gcloudb", "gcloudub", "checkerboard"]
large_data_list = ["spambase", "banana", "phoneme", "ringnorm", "twonorm", "phishing"]
data_list = small_data_list + large_data_list


# ## Align results
# 
# We align all results on
# - small datasets $n < 2000$ : more than 100 indicis
# - large datasets $n \geq 2000$ : more than 10 indicis.

# In[3]:


names = os.listdir('./aubc/')
qs_map_pos = {k: i for i, k in enumerate(qs_list)}
table3_idx = {k: [None for _ in range(len(qs_list))] for k in data_list}


# In[4]:


for name in names:
    if not name.endswith('.csv'):
        continue
    terms = name.split('-')
    if 'look' in name:
        qs = terms[1] + terms[6].split('_')[-1][-4:]
    else:
        qs = terms[1]

    data = terms[0]

    res = pd.read_csv(os.path.join('./aubc/', name))

    idx = res['res_expno'].unique()

    if data in large_data_list:
        if len(idx) < 10:
            print(f'{data}-{qs}: {len(idx)} < 10 times')
            continue
    else:
        if len(idx) < 100:
            print(f'{data}-{qs}: {len(idx)} < 100 times')
            continue

    table3_idx[data][qs_map_pos[qs]] = idx


# In[5]:


def align_idx(idxArr_list):
    res = idxArr_list[0]
    for idxArr in idxArr_list[1:]:
        if idxArr is None:
            continue

        res = np.intersect1d(res, idxArr)

    return res

aligned_idx = []
for data in table3_idx:
    align_idx_arr = align_idx(table3_idx[data])
    if data in large_data_list:
        align_idx_arr = align_idx_arr[:10]
        assert align_idx_arr.shape[0] == 10, f'Size of {data} is not correct. {(align_idx_arr.shape[0])}'
    else:
        align_idx_arr = align_idx_arr[:100]
        assert align_idx_arr.shape[0] == 100, f'Size of {data} is not correct. {(align_idx_arr.shape[0])}'

    n_exp = len(align_idx_arr)
    aligned_idx.append([data, n_exp, f'{align_idx_arr.tolist()}'])

aligned_idx = pd.DataFrame(aligned_idx)


# In[6]:


aligned_idx_dict = {}
for data, idx in zip(aligned_idx[0], aligned_idx[2]):
    aligned_idx_dict[data] = eval(idx)


# # Reproducing Zhan et al. Results.

# ## Table. Summary Table
# 
# - small datasets $n < 2000$ : only use first 100 indicis $K_{S} = 100$.
# - large datasets $n \geq 2000$ : only use first 10 indicis $K_{L} = 10$.
# 
# Calculate average (mean) and standard deviation of AUBCs by
# $$
# \overline{\mathrm{AUBC}}_{q, s} = \frac{\sum_{k=1}^{K_{\bullet}} \mathrm{AUBC}_{q, s, k}}{K_{\bullet}},
# $$
# where $K_{\bullet} \in {K_{S}, K_{L}}$.

# In[7]:


mean_aubc_q_s = []
std_aubc_q_s = []
index_duplicate = []
for name in names:
    if not name.endswith('.csv'):
        continue

    terms = name.split('-')
    if 'look' in name:
        qs = terms[1] + terms[6].split('_')[-1][-4:]
    else:
        qs = terms[1]

    data = terms[0]

    if (data, qs) in index_duplicate:
        breakpoint()

    index_duplicate.append((data, qs))
    res = pd.read_csv(os.path.join('./aubc/', name))

    # aligned index
    if aligned_idx_dict is not None:
        res = res[res['res_expno'].isin(aligned_idx_dict[data])]

    cnt_aubc = res['res_tst_score'].count()
    if data in large_data_list:
        if cnt_aubc < 10:
            continue
    else:
        if cnt_aubc < 100:
            continue

    mean_aubc_ = res['res_tst_score'].mean()
    std_aubc_ = res['res_tst_score'].std()
    mean_aubc_ = round(mean_aubc_, 4)
    std_aubc_ = round(std_aubc_, 4)

    mean_aubc_q_s.append([data, qs, mean_aubc_])
    std_aubc_q_s.append([data, qs, std_aubc_])


# In[8]:


mean_aubc_q_s = pd.DataFrame(mean_aubc_q_s)
std_aubc_q_s = pd.DataFrame(std_aubc_q_s)

mean_aubc_q_s.columns = ['data', 'qs', 'aubc_mean']
std_aubc_q_s.columns = ['data', 'qs', 'aubc_std']

mean_aubc_q_s = pd.pivot(mean_aubc_q_s, values='aubc_mean', index=['qs'], columns=['data'])
std_aubc_q_s = pd.pivot(std_aubc_q_s, values='aubc_std', index=['qs'], columns=['data'])


# In[9]:


mean_aubc_q_s = mean_aubc_q_s.reindex(index=qs_list, columns=data_list)
std_aubc_q_s = std_aubc_q_s.reindex(index=qs_list, columns=data_list)


# In[10]:


# str of mean and std AUBCs
mean_aubc_q_s_str = mean_aubc_q_s.copy().astype(str)
std_aubc_q_s_str = std_aubc_q_s.copy().astype(str)
for d in mean_aubc_q_s.columns:
    bst_q = mean_aubc_q_s.loc[al_list, d].nlargest(3+1).index  # as margin == infodiv in the current setting
    if 'infodiv' not in bst_q:
        bst_q = bst_q[:3]
    else:
        bst_q = bst_q.drop('infodiv')

    # export to GitHub
    mean_aubc_q_s_str.loc[bst_q[0], d] = f'{mean_aubc_q_s_str.loc[bst_q[0], d]}¹'
    mean_aubc_q_s_str.loc[bst_q[1], d] = f'{mean_aubc_q_s_str.loc[bst_q[1], d]}²'
    mean_aubc_q_s_str.loc[bst_q[2], d] = f'{mean_aubc_q_s_str.loc[bst_q[2], d]}³'

    bst_q_std = std_aubc_q_s.loc[al_list, d].nsmallest(3+1).index
    if 'infodiv' not in bst_q_std:
        bst_q_std = bst_q_std[:3]
    else:
        bst_q_std = bst_q_std.drop('infodiv')

    # export to GitHub
    std_aubc_q_s_str.loc[bst_q_std[0], d] = f'{std_aubc_q_s_str.loc[bst_q_std[0], d]}¹'
    std_aubc_q_s_str.loc[bst_q_std[1], d] = f'{std_aubc_q_s_str.loc[bst_q_std[1], d]}²'
    std_aubc_q_s_str.loc[bst_q_std[2], d] = f'{std_aubc_q_s_str.loc[bst_q_std[2], d]}³'


# In[11]:


mean_std_aubc_q_s_str = mean_aubc_q_s_str + '(' + std_aubc_q_s_str + ')'
mean_std_aubc_q_s_str = mean_std_aubc_q_s_str.loc[qs_list, :]
mean_std_aubc_q_s_str.index = ['uniform', 'us', 'qbc', 'hintsvm', 'quire', 'albl', 'dwus', 'vr',
                               'kcenter', 'margin', 'graph', 'hier', 'infodiv', 'mcm', 'eer', 'bmdr',
                               'spal', 'lal', 'bso']
mean_std_aubc_q_s_str = mean_std_aubc_q_s_str.T
mean_std_aubc_q_s_str = mean_std_aubc_q_s_str.replace(to_replace='nan(nan)', value='too long (time)')
mean_std_aubc_q_s_str.loc['checkerboard', 'spal'] = 'error'
mean_std_aubc_q_s_str.loc['spambase', 'quire'] = 'error'


# In[12]:


summary_table = mean_std_aubc_q_s_str.to_markdown()
summary_table = '# Benchmark of pool-based active learning\n\nMean(Standard Deviation) of Uniform (Random Sampling), 17 query strategies and Beam-Search Oracle (BSO) on 26 binary datasets.\n\n' + summary_table
with open('./README.md', 'w') as f:
    f.write(summary_table)


# ## Re-Benchmark of Table 3 in Zhan et al.
# 
# - RS (Uniform): $\overline{\mathrm{AUBC}}_{q=\text{Uniform}, s}$.
# - BSO: $\overline{\mathrm{AUBC}}_{q=\text{BSO}, s}$.
# - Avg: average of $17$ query strategies.
# $$
# \overline{\mathrm{AUBC}}_{s} = \frac{\sum_{q \in \text{qs}} \overline{\mathrm{AUBC}}_{q, s}}{17}
# $$
# where $\text{qs} = $ {'us', 'qbc', 'hintsvm', 'quire', 'albl', 'dwus', 'vr', 'kcenter', 'margin', 'graph', 'hier', 'infodiv', 'mcm', 'eer', 'bmdr', 'spal', 'lal'}
# - BEST_val: $\max_{q} \overline{\mathrm{AUBC}}_{q, s}$ and BEST_mhd: $\arg\max_{q} \overline{\mathrm{AUBC}}_{q, s}$
# - WORST_val: $\min_{q} \overline{\mathrm{AUBC}}_{q, s}$ and WORST_mhd: $\arg\min_{q} \overline{\mathrm{AUBC}}_{q, s}$
# 
# We also check whether mean of AUBCs in [Zhan et al., 2021] locating in
# - confidence interval with $\alpha=0.05$ significance level.
# - confidence interval with $\alpha=0.01$ significance level.
# 
# We suppose both of experiments have the same settings.
# They will generate independent, identical distribution (i.i.d.) results.
# 
# *ChatGPT*
# > If you have the mean of one sample and you want to compare it to the median of another sample, you can use the confidence interval for the mean of the first sample to see if the median of the second sample falls within the interval. This will give you an idea of whether the median of the second sample is significantly different from the mean of the first sample, but it will not be the same as the Mann-Whitney U test, which compares the medians of two independent samples.
# > This will calculate the 95% confidence interval for the mean of the first sample. You can then compare the median of the second sample to this interval to see if it falls within the interval. If the median falls within the interval, it suggests that the median is not significantly different from the mean of the first sample. If the median falls outside the interval, it suggests that the median is significantly different from the mean of the first sample.
# > Keep in mind that this approach will give you an idea of whether the median of the second sample is significantly different from the mean of the first sample, but it will not provide a formal hypothesis test or p-value like the Mann-Whitney U test.

# In[13]:


# Add XZ2021 results
xz2021_table3 = pd.read_csv('table3-xz2021.csv')
# align to our code
xz2021_table3 = xz2021_table3.set_index('XZ2021')
xz2021_table3.columns = ['uniform', 'bsoDtst', 'Avg', 'BEST_val', 'BEST_mhd', 'WORST_val', 'WORST_mhd']


# ### Table. Reporducing Failure of Uniform
# 
# Check the difference between Zhan et al. and ours on Uniform

# In[14]:


table3_uniform = pd.DataFrame()  # our results
table3_uniform.loc[:, 'mean'] = mean_aubc_q_s.loc['uniform', :]
table3_uniform.loc[:, 'SD'] = std_aubc_q_s.loc['uniform', :]
table3_uniform.loc[:, '\cite{XZ2021}'] = xz2021_table3['uniform']
table3_uniform.loc[:, '$\\alpha=5\%$'] = None
table3_uniform.loc[:, '$\\alpha=1\%$'] = None
table3_uniform.index.name = f'{table3_uniform.index.name}($\%$)'

def tinterval_check(mean_poy, std_poy, n_poy, mean_XZ2021):
    se = std_poy / np.sqrt(n_poy)
    ci_95 = stats.t.interval(alpha=0.95, df=n_poy-1, loc=mean_poy, scale=se)
    if ci_95[0] <= mean_XZ2021 <= ci_95[1]:
        decision_95 = 0  # not significantly different with 95 confidence interval
    else:
        decision_95 = 1  # significantly different with 95 confidence interval

    ci_99 = stats.t.interval(alpha=0.99, df=n_poy-1, loc=mean_poy, scale=se)
    if ci_99[0] <= mean_XZ2021 <= ci_99[1]:
        decision_99 = 0  # not significantly different with 95 confidence interval
    else:
        decision_99 = 1  # significantly different with 95 confidence interval

    return decision_95, decision_99

for data_name in table3_uniform.index:
    if data_name in large_data_list:
        n_samples = 10
    else:
        n_samples = 100

    d_95, d_99 = tinterval_check(
        table3_uniform.loc[data_name, 'mean'],
        table3_uniform.loc[data_name, 'SD'],
        n_samples,
        table3_uniform.loc[data_name, '\cite{XZ2021}']
    )

    if d_95 == 1:
        table3_uniform.loc[data_name, '$\\alpha=5\%$'] = f'Out'
    else:
        table3_uniform.loc[data_name, '$\\alpha=5\%$'] = f'In'

    if d_99 == 1:
        table3_uniform.loc[data_name, '$\\alpha=1\%$'] = f'Out'
    else:
        table3_uniform.loc[data_name, '$\\alpha=1\%$'] = f'In'

table3_uniform['mean'] = table3_uniform['mean'].apply(lambda x: f'{x:.2%}'[:-1])
table3_uniform['SD'] = table3_uniform['SD'].apply(lambda x: f'{x:.2%}'[:-1])
table3_uniform['\cite{XZ2021}'] = table3_uniform['\cite{XZ2021}'].apply(lambda x: f'{x:.1%}'[:-1])


# In[15]:


table3_uniform.to_latex('rsfail.tex',
                        label='tab2:rsfail',
                        caption='Reporducing Failure of \\textbf{Uniform}',
                        escape=False)


# ### Reporducing Failure of BSO
# 
# Check the difference between Zhan et al. and ours on BSO

# In[16]:


table3_bso = pd.DataFrame()
table3_bso.loc[:, 'mean'] = mean_aubc_q_s.loc['bsoDtst', :]
table3_bso.loc[:, 'SD'] = std_aubc_q_s.loc['bsoDtst', :]
table3_bso.loc[:, '\cite{XZ2021}'] = xz2021_table3['bsoDtst']
table3_bso.loc[:, '$\\alpha=5\%$'] = None
table3_bso.loc[:, '$\\alpha=1\%$'] = None
table3_bso.index.name = f'{table3_bso.index.name}($\%$)'

n_samples = 100
for data_name in table3_bso.index:
    if data_name in large_data_list:
        continue

    d_95, d_99 = tinterval_check(
        table3_bso.loc[data_name, 'mean'],
        table3_bso.loc[data_name, 'SD'],
        n_samples,
        table3_bso.loc[data_name, '\cite{XZ2021}']
    )

    if d_95 == 1:
        table3_bso.loc[data_name, '$\\alpha=5\%$'] = f'Out'
    else:
        table3_bso.loc[data_name, '$\\alpha=5\%$'] = f'In'

    if d_99 == 1:
        table3_bso.loc[data_name, '$\\alpha=1\%$'] = f'Out'
    else:
        table3_bso.loc[data_name, '$\\alpha=1\%$'] = f'In'

table3_bso['mean'] = table3_bso['mean'].apply(lambda x: f'{x:.2%}'[:-1])
table3_bso['SD'] = table3_bso['SD'].apply(lambda x: f'{x:.2%}'[:-1])
table3_bso['\cite{XZ2021}'] = table3_bso['\cite{XZ2021}'].apply(lambda x: f'{x:.1%}'[:-1])
table3_bso = table3_bso.iloc[:-6, :]


# In[17]:


table3_bso.to_latex('bsofail.tex',
                    label='tab2:bsofail',
                    caption='Reporducing Failure of \\textbf{BSO}',
                    escape=False)


# ### Re-Benchmarking all results of Table 3 in Zhan et al.

# In[18]:


table3 = xz2021_table3.copy().applymap(lambda x: None)


# In[19]:


# 1. deal with uniform and bso
for qs_name in ['uniform', 'bsoDtst']:
    for data_name in table3.index:
        if data_name in large_data_list:
            n_samples = 10
        else:
            n_samples = 100

        if np.isnan(xz2021_table3.loc[data_name, qs_name]):
            continue
            
        d_95, d_99 = tinterval_check(
            mean_aubc_q_s.loc[qs_name, data_name],
            std_aubc_q_s.loc[qs_name, data_name],
            n_samples,
            xz2021_table3.loc[data_name, qs_name]
        )

        # update value with Poy's results
        table3.loc[data_name, qs_name] = f'{mean_aubc_q_s.loc[qs_name, data_name]:.2%}'[:-1]

        # show results
        if d_95 == 1:
            table3.loc[data_name, qs_name] = f'{table3.loc[data_name, qs_name]}*'

        # if d_99 == 1:
        #     report4.loc[data_name, qs_name] = f'{report4.loc[data_name, qs_name]}*'


# In[20]:


# 2. deal with Avg
table3_avg = mean_aubc_q_s.loc[al_list, :].mean().round(4)
table3_avg_std = mean_aubc_q_s.loc[al_list, :].std().round(6)
table3_avg_cnt = len(al_list)
col = 'Avg'
for data_name in table3.index:
    if np.isnan(xz2021_table3.loc[data_name, col]):
        continue

    d_95_avg, d_99_avg = tinterval_check(
        table3_avg.loc[data_name],
        table3_avg_std.loc[data_name],
        table3_avg_cnt,
        xz2021_table3.loc[data_name, col]
    )

    # update value with Poy's results
    table3.loc[data_name, col] = f'{table3_avg.loc[data_name]:.2%}'[:-1]

    # show results
    if d_95_avg == 1:
        table3.loc[data_name, col] = f'{table3.loc[data_name, col]}*'

    # if d_99_avg == 1:
    #     report4.loc[data_name, col] = f'{report4.loc[data_name, col]}*'


# In[21]:


# 3. deal with BEST and WORST results
for data_name in table3.index:
    # update value with Poy's results
    qs_name = mean_aubc_q_s.loc[al_list, data_name].idxmax()
    table3.loc[data_name, 'BEST_val'] = f'{mean_aubc_q_s.loc[qs_name, data_name]:.2%}'[:-1]
    if xz2021_table3.loc[data_name, 'BEST_mhd'] != qs_name:
        table3.loc[data_name, 'BEST_mhd'] = f'{qs_name}*'
    else:
        table3.loc[data_name, 'BEST_mhd'] = qs_name

for data_name in table3.index:
    # update value with Poy's results
    qs_name = mean_aubc_q_s.loc[al_list, data_name].idxmin()
    table3.loc[data_name, 'WORST_val'] = f'{mean_aubc_q_s.loc[qs_name, data_name]:.2%}'[:-1]
    if xz2021_table3.loc[data_name, 'WORST_mhd'] != qs_name:
        table3.loc[data_name, 'WORST_mhd'] = f'{qs_name}*'
    else:
        table3.loc[data_name, 'WORST_mhd'] = qs_name


# In[22]:


# 4. final results
table3_latex = table3.copy().fillna('-')
table3_latex.index.name = 'data ($\%$)'
table3_latex.columns = ['RS', 'BSO', 'Avg', 'BEST\\_val', 'BEST\\_mhd', 'WORST\\_val', 'WORST\\_mhd']


# In[23]:


table3_latex_str = table3_latex.to_latex(
    label='tab2:tab3',
    caption='Re-Benchmarking all results of Table 3 in \\citep{XZ2021}',
    escape=False
)
table3_latex_str = table3_latex_str.replace('{table}', '{table*}')  # cross 2 columns
with open('table2-table3.tex', 'w') as f:
    f.write(table3_latex_str)


# ## Re-Benchmark of Table 4 in Zhan et al.
# 
# We report the **dimension**, **scale**, **imbalance ratio** aspects as [Zhan et al., 2021].
# 
# - **dimension**: Low-Dimension ($d < 50$), High-Dimensio ($d \geq 50$)
# - **scale**: Small-Scale ($n < 1000$), Large-Scale ($n \geq 1000$)
# - **imbalance ratio**: BALance ($r < 1.5$), IMBalance ($r \geq 1.5$)
# 
# > We present the average performance difference between the best AL/BSO and the AL method,
# > i.e., $\delta_{i} = \max(\text{BSO}, a_{1}, \dots, a_{17}) - a_{i}$,
# > where $a_{i}$ is the AUBC for the $i$-th method.
# 

# In[24]:


# load table 2
xz2021_table2 = pd.read_csv('table2-xz2021.csv', index_col=0)
xz2021_table2 = xz2021_table2.drop(['fourclass'], axis=1)
xz2021_table2.columns = ['appendicitis', 'sonar', 'parkinsons', 'ex8b', 'heart', 'haberman',
                         'ionosphere', 'clean1', 'breast', 'wdbc', 'australian', 'diabetes',
                         'mammographic', 'ex8a', 'tic', 'german', 'splice', 'gcloudb',
                         'gcloudub', 'checkerboard', 'spambase', 'banana', 'phoneme', 'ringnorm',
                         'twonorm', 'phishing']
xz2021_table2 = xz2021_table2.T
xz2021_table2['d'] = xz2021_table2['d'].astype(int)
xz2021_table2['n'] = xz2021_table2['n'].astype(int)
xz2021_table2['K'] = xz2021_table2['K'].astype(int)


# In[25]:


# categories of dimension, scale and imbalance ratio
Bin_data = xz2021_table2[xz2021_table2['K']==2].index
Mul_data = xz2021_table2[xz2021_table2['K']>2].index
LD_data = xz2021_table2[xz2021_table2['d']<50].index
HD_data = xz2021_table2[xz2021_table2['d']>=50].index
SS_data = xz2021_table2[xz2021_table2['n']<1000].index
LS_data = xz2021_table2[xz2021_table2['n']>=1000].index
Sy_data = ['ex8b', 'ex8a', 'gcloudb', 'gcloudub', 'checkerboard', 'banana']
Re_data = [data for data in xz2021_table2.index.tolist() if data not in Sy_data]
BAL_data = xz2021_table2[xz2021_table2['IR']<1.5].index
IMB_data = xz2021_table2[xz2021_table2['IR']>=1.5].index


# In[26]:


# delta
qs_list_wo_uniform = qs_list[1:]
table4_mean = mean_aubc_q_s.loc[qs_list_wo_uniform, :]
table4_delta = table4_mean.max() - table4_mean
table4_delta = table4_delta.loc[al_list, :]
table4_delta.head()


# ### Table. Verifying Applicability with $\delta_{i}$

# In[27]:


# summary for table 4
all_ = table4_delta.mean(axis=1)
binary = table4_delta[Bin_data].mean(axis=1)
multi = table4_delta[Mul_data].mean(axis=1)
LD = table4_delta[LD_data].mean(axis=1)
HD = table4_delta[HD_data].mean(axis=1)
small = table4_delta[SS_data].mean(axis=1)
large = table4_delta[LS_data].mean(axis=1)
synt = table4_delta[Sy_data].mean(axis=1)
real = table4_delta[Re_data].mean(axis=1)
bal = table4_delta[BAL_data].mean(axis=1)
imbal = table4_delta[IMB_data].mean(axis=1)
table4 = pd.concat([all_, binary, multi, LD, HD, small, large, synt, real, bal, imbal], axis=1)
table4.columns = ['All', 'B', 'M', 'LD', 'HD', 'SS', 'LS', 'R', 'S', 'BAL', 'IMB']
table4 = table4.round(4)


# In[28]:


table4_latex = table4.loc[:, ['B', 'LD', 'HD', 'SS', 'LS', 'BAL', 'IMB']]
for col in table4_latex.columns:
    col_nsmallest = table4_latex.nsmallest(4, col, keep='all')  # (margin == infodiv) when query batch size = 1
    rank_qs = col_nsmallest.index
    # export to LaTeX format
    table4_latex.loc[:, col] = table4_latex.loc[:, col].apply(lambda x: f'{x:.2%}'[:-1])
    # add rank of the best 3 methods
    for rank, qs in enumerate(rank_qs):
        if rank == 0:
            rank = 1
        table4_latex.loc[qs, col] = f'{table4_latex.loc[qs, col]}\\textsuperscript{{{rank}}}'

table4_latex.index.name = f'{table4_latex.index.name}($\%$)'


# In[29]:


table4_latex.to_latex(
    'table3-table4.tex',
    label='tab3:tab4',
    caption='Verifying Applicability with $\\delta_{i}$',
    escape=False
)


# ## Revision of Table 2 in Zhan et al.

# In[30]:


# Export Table 2 to LaTeX
xz2021_table2_report = pd.read_csv('table2-xz2021Report.csv', sep='|')  # copy from the PDF
xz2021_table2_report.columns = ['Dataset', 'Property', 'IR', '(d, n, K)']
xz2021_table2_report['d'] = xz2021_table2_report['(d, n, K)'].str.strip().str[1:-1].str.split(',').str[0].astype(int)
xz2021_table2_report['n'] = xz2021_table2_report['(d, n, K)'].str.strip().str[1:-1].str.split(',').str[1].astype(int)
xz2021_table2_report['K'] = xz2021_table2_report['(d, n, K)'].str.strip().str[1:-1].str.split(',').str[2].astype(int)
xz2021_table2_report = xz2021_table2_report.drop('(d, n, K)', axis=1)
xz2021_table2_report = xz2021_table2_report.set_index('Dataset')
xz2021_table2_report.index = ['appendicitis', 'sonar', 'iris', 'wine', 'parkinson', 'ex8b',
       'seeds', 'glass', 'thyroid', 'heart', 'haberman', 'ionosphere',
       'clean1', 'breast', 'wdbc', 'r15', 'australian',
       'diabetes', 'mammographic', 'ex8a', 'vehicle',
       'tic', 'german', 'splice',
       'gcloudb', 'gcloudub',
       'checkerboard', 'phishing', 'd31', 'spambase',
       'banana', 'phoneme', 'texture', 'ringnorm', 'twonorm']


# In[31]:


table2 = pd.merge(
    xz2021_table2, xz2021_table2_report,
    how='left', left_index=True, right_index=True,
    suffixes=('', '_xz2021')
).dropna()


# In[32]:


table2_latex = table2.copy()
table2_latex['d_str'] = table2_latex.apply(lambda x: str(int(x['d_xz2021'])) + '$\\rightarrow$' + str(x['d'])
                                       if x['d'] != x['d_xz2021']
                                       else str(int(x['d'])), axis=1
                                      )
table2_latex['IR_str'] = table2_latex.apply(lambda x: str(int(x['IR_xz2021'])) + '$\\rightarrow$' + str(x['IR'])
                                       if x['IR'] != x['IR_xz2021']
                                       else str(int(x['IR'])), axis=1
                                      )
table2_latex['n_str'] = table2_latex.apply(lambda x: str(int(x['n_xz2021'])) + '$\\rightarrow$' + str(x['n'])
                                       if x['n'] != x['n_xz2021']
                                       else str(int(x['n'])), axis=1
                                      )
table2_latex['K_str'] = table2_latex.apply(lambda x: str(int(x['K_xz2021'])) + '$\\rightarrow$' + str(x['K'])
                                       if x['K'] != x['K_xz2021']
                                       else str(int(x['K'])), axis=1
                                      )

table2_latex = table2_latex[['Property', 'IR_str', 'd_str', 'n_str', 'K_str']]
table2_latex.columns = ['Property', '$r$', '$d$', '$n$', '$K$']
table2_latex = table2_latex[table2_latex.columns[:-1]]


# In[33]:


table2_latex.to_latex(
    'table2-table2.tex',
    label='tab2:tab2',
    caption='Revision of Table 2 in \\citep{XZ2021}',
    escape=False
)


# # Proposed Analysis Methods

# ## Figure. The learning curves of query strategies on Heart
# 
# Plot learning curves of several query strategies on *Heart* dataset.
# - Input. `detail/*.csv` with format "seed|round|accuracy|time of training|time of querying".
# - Align the seed of query strategies.
# - Calculate mean and standard deviation (SD) of accuracy.
# - Plot mean, upper bound (mean + SD) and lower bound (mean - SD) of accuracy (y-axis) along number of labels (x-axis).

# In[34]:


# Please change data and (lc_qs, colors) by yourself.
data = 'heart'
lc_qs = [
    'uniform-zhan-google-zhan',  # gray
    'us-zhan-us-zhan',           # r
    'qbc-zhan-qbc-zhan',         # r
    'albl-zhan-albl-zhan',       # r
    'kcenter-zhan-libact-zhan',  # g
    'margin-zhan-google-zhan',   # r
    'mcm-zhan-google-zhan',      # b
    'eer-zhan-eer-zhan',       # r
    'spal-zhan-alipy-zhan',      # b
    'lal-zhan-alipy-zhan',       # b
]
linestys = ['-']
colors = ['gray', 'gold', 'orange', 'deeppink',
          'green', 'red', 'blue',
          'brown', 'steelblue', 'violet']


# In[35]:


def read_detail_csv(name, align=None):
    path = Path(name)
    if not path.is_file():
        return None

    df = pd.read_csv(name, header=None, sep='|')
    df = df.loc[:, [0, 1, 2]]  # 0: seed, 1: round, 2: test accuracy
    if align:
        df = df[df[0].isin(align)]
    else:
        pass

    df = df.dropna()
    return df

def clean_detail_df(df):
    try:
        if df[2].dtype == float:
            pass
        else:
            df = df[df[2].str.contains('lr')==False]
            df[2] = df[2].astype(float)
    except:
        df = None

    return df

learning_curves = []
skip_qs_idx = []
for i, exp in enumerate(lc_qs):
    res_detail = read_detail_csv(f'detail/{data}-{exp}-zhan-RS_noFix_scale-detail.csv', align=aligned_idx_dict[data])
    clean_detail = clean_detail_df(res_detail)
    if clean_detail is None:
        print(f'{name} fail')
        import pdb; pdb.set_trace()
        skip_qs_idx.append(i)                                                                                                                                            
        continue

    if clean_detail.shape[1] > 5:
        clean_detail = clean_detail_df(clean_detail)
        clean_detail.columns = range(5)

    res_lc = clean_detail.groupby(1).agg({2: ['mean', 'std', 'count']})
    res_lc.columns = ['avg', 'std', 'cnt']
    res_lc.index.name = None
    res_lc.index = res_lc.index.astype(int)
    learning_curves.append(res_lc)

lc_qs = [x for i, x in enumerate(lc_qs) if i not in skip_qs_idx]
colors = [x for i, x in enumerate(colors) if i not in skip_qs_idx]


# In[36]:


avgs = pd.concat([df['avg'] for df in learning_curves], axis=1)
stds = pd.concat([df['std'] for df in learning_curves], axis=1)
avgs.columns = lc_qs
stds.columns = lc_qs
upper = avgs + stds
lower = avgs - stds


# In[37]:


fig = plt.figure(figsize=(12,8))
for i, name in enumerate(lc_qs):
    if name not in ('uniform-zhan-google-zhan', 'margin-zhan-google-zhan', 'mcm-zhan-google-zhan'):
        continue

    cur_line = avgs[name].dropna()
    plt.plot(cur_line.index, cur_line, linestyle=linestys[0], color=colors[i], label=name)
    cur_upper = upper[name].dropna()
    cur_lower = lower[name].dropna()
    plt.fill_between(cur_lower.index, cur_lower, cur_upper, facecolor=colors[i], alpha=0.1)

plt.xlabel("# of labeled samples")
plt.ylabel("Test Accuracy")
plt.legend(ncol=2, loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
export_name = f"lc-{data}.png"
plt.savefig(export_name, bbox_inches='tight')                                                                                                                       
plt.clf()


# ## Figure. Difference AUBC between Margin and Uniform on *Heart*
# 
# Plot learning curves of several query strategies on *Heart* dataset.
# - Input. `aubc/*.csv` with format "res_expno,res_lbl_score,res_tst_score".
# - Align the seed of query strategies.
# - Calculate difference AUBC between {qs} and Uniform.
# - Plot difference AUBC (y-axis) along index of seeds (x-axis).

# In[38]:


data = 'heart'
qs = [
    'uniform-zhan-google-zhan',  # Uniform
    'margin-zhan-google-zhan',
]


# In[39]:


def read_aubc_csv(name, align=None):
    path = Path(name)
    if not path.is_file():
        return None

    df = pd.read_csv(name)
    if align:
        df = df[df['res_expno'].isin(align)]
    else:
        pass

    df = df.dropna()
    df = df.drop_duplicates()
    df = df.set_index('res_expno')
    df = df.loc[:, 'res_tst_score']
    if df.shape[0] not in (100, 10):
        print(f'{name} Error #exps {df.shape[0]}!')
        df = None

    return df

aubc_qs_data_seed = []
for i, exp in enumerate(qs):
    res_aubc = read_aubc_csv(f'aubc/{data}-{exp}-zhan-RS_noFix_scale-aubc.csv', align=aligned_idx_dict[data])
    if res_aubc is None:
        print(f'{exp} fail')
        import pdb; pdb.set_trace()
        skip_qs_idx.append(i)                                                                                                                                            
        continue

    aubc_qs_data_seed.append(res_aubc)


# In[40]:


tau_qs_data_seed = (aubc_qs_data_seed[1] - aubc_qs_data_seed[0]).to_frame()


# In[41]:


tau_qs_data_seed_mean = tau_qs_data_seed['res_tst_score'].mean()
tau_qs_data_seed_std = tau_qs_data_seed['res_tst_score'].std()
tau_qs_data_seed_cnt = tau_qs_data_seed['res_tst_score'].count()
tau_qs_data_seed_sem = tau_qs_data_seed_std/np.sqrt(tau_qs_data_seed_cnt)


# In[42]:


plt.rcParams["figure.figsize"] = (12, 8)
tau_qs_data_seed['res_tst_score'].plot(marker='.', linestyle='--', c='tab:purple', label='margin', alpha=0.5, ms=10)
plt.xlabel('Index of Seed')
plt.ylabel('Difference AUBC between Margin from Uniform')
plt.axhline(y=0, linestyle='--', color='black')
plt.axhline(y=tau_qs_data_seed_mean, color='tab:purple', label='mean difference AUBC')
plt.gca().add_patch(plt.Rectangle((0, tau_qs_data_seed_mean-tau_qs_data_seed_sem), 100, 2*tau_qs_data_seed_sem, facecolor="orange", alpha=0.5, label='95% C.I.'))
plt.legend()
plt.savefig('scat-improve.png', bbox_inches='tight')
plt.clf()


# ## Table. (Online) $\bar{\tau}_{q, s}$ and $\text{SD}(\tau)_{q, s}$
# 
# Estimate the mean and SD of $\tau_{q, s}$ by seeds and illustrate $95\%$ confidence interval (C.I.).
# 
# - Input. `aubc/*.csv` with format "res_expno,res_lbl_score,res_tst_score".
# - Align the seed of query strategies.
# - Calculate difference AUBC between {qs} and Uniform.
# - Calculate mean and SD of difference AUBC.

# In[43]:


qs_file_list = ['uniform-zhan-google-zhan', 'us-zhan-us-zhan', 'qbc-zhan-qbc-zhan', 'hintsvm-zhan-libact-zhan', 'quire-zhan-libact-zhan',
                'albl-zhan-albl-zhan', 'dwus-zhan-libact-zhan', 'vr-zhan-vr-zhan', 'kcenter-zhan-libact-zhan',  # libact
                'margin-zhan-google-zhan', 'graph-zhan-google-zhan', 'hier-zhan-google-zhan', 'infodiv-zhan-google-zhan', 'mcm-zhan-google-zhan',  # google
                'eer-zhan-eer-zhan', 'bmdr-zhan-alipy-zhan', 'spal-zhan-alipy-zhan', 'lal-zhan-alipy-zhan',  # alipy
                'bso-zhan-bso-zhan']

aubc_qs_data_seed = {}
aubc_data_skipQS = {}
for qs in qs_file_list:
    for data in data_list:
        if 'bso' not in qs:
            res_aubc = read_aubc_csv(f'aubc/{data}-{qs}-zhan-RS_noFix_scale-aubc.csv', align=aligned_idx_dict[data])
        else:
            res_aubc = read_aubc_csv(f'aubc/{data}-{qs}-zhan-RS_noFix_scale_lookDtst-aubc.csv', align=aligned_idx_dict[data])

        if aubc_qs_data_seed.get(data):
            aubc_qs_data_seed[data].update({qs: res_aubc})
        else:
            aubc_qs_data_seed[data] = {qs: res_aubc}


# In[44]:


tau_qs_data_seed = {}
for data in aubc_qs_data_seed:
    aubc_ = pd.DataFrame(aubc_qs_data_seed[data])
    aubc_.columns = qs_list
    tau_ = aubc_.sub(aubc_['uniform'], axis=0)
    tau_qs_data_seed[data] = tau_


# In[45]:


tau_qs_data_mean = {data: tau_qs_data_seed[data].mean() for data in tau_qs_data_seed}
tau_qs_data_mean = pd.DataFrame(tau_qs_data_mean)
tau_qs_data_mean = tau_qs_data_mean.loc[al_list, :]
tau_qs_data_mean = tau_qs_data_mean.loc[:, data_list]


# In[46]:


tau_qs_data_std = {data: tau_qs_data_seed[data].std() for data in tau_qs_data_seed}
tau_qs_data_std = pd.DataFrame(tau_qs_data_std)
tau_qs_data_std = tau_qs_data_std.loc[al_list, :]
tau_qs_data_std = tau_qs_data_std.loc[:, data_list]


# In[47]:


not_enough_exps = []  # normaltest
sim_performance_datasets = []  # bartlett
tau_data_qs = tau_qs_data_mean.copy().applymap(lambda x: None)
tau_data_qs_cnt = tau_data_qs.copy()
for data in aubc_qs_data_seed:
    aubc_ = pd.DataFrame(aubc_qs_data_seed[data])
    aubc_.columns = qs_list
    aubc_ = aubc_.dropna(axis=1)
    aubc_uniform_ = aubc_['uniform']
    aubc_qss_ = aubc_.drop(['uniform'], axis=1)
    if 'bsoDtst' in aubc_qss_:
        aubc_qss_ = aubc_qss_.drop(['bsoDtst'], axis=1)

    # hypothesis test
    # check for normality of each qs (column)
    # https://www.allendowney.com/blog/2023/01/28/never-test-for-normality/
    # Results of normality test: not enough of experiments
    alpha = 1e-3
    cur_data_qss = []
    for qs in aubc_qss_.columns:
        _, p_val = stats.normaltest(aubc_qss_[qs].values)
        if p_val < alpha:  # null hypothesis: x comes from a normal distribution
            not_enough_exps.append((qs, data))
        else:
            pass

        cur_data_qss.append(aubc_qss_[qs])
        # mean of difference between (Uniform, QS)
        cur_tau = aubc_qss_[qs].sub(aubc_uniform_, axis=0).mean().round(4)
        tau_data_qs.loc[qs, data] = cur_tau
        tau_data_qs_cnt.loc[qs, data] = cur_tau  # Use for judge = or < uniform

    # check for equal variances of all qs (data)
    _, p_val = stats.bartlett(*cur_data_qss)
    if p_val < alpha:  # null hypothesis: x comes from a normal distribution
        sim_performance_datasets.append(data)
    else:
        pass

    # paired t-test of each qs (column)
    for qs in aubc_qss_.columns:
        _, p_value = stats.ttest_rel(aubc_uniform_, aubc_qss_[qs], alternative='less')
        alpha_95 = 0.05
        alpha_99 = 0.01

        # update table with decision
        if p_value < alpha_95:  # > uniform with 95 CI
            tau_data_qs.loc[qs, data] = f'{tau_data_qs.loc[qs, data]}*'
            tau_data_qs_cnt.loc[qs, data] = 1
        if p_value < alpha_99:  #  # > uniform with 99 CI
            tau_data_qs.loc[qs, data] = f'{tau_data_qs.loc[qs, data]}*'
            tau_data_qs_cnt.loc[qs, data] = 2
        else:
            if tau_data_qs_cnt.loc[qs, data] < 0:  # < uniform
                tau_data_qs_cnt.loc[qs, data] = -1
            else:  # = uniform
                tau_data_qs_cnt.loc[qs, data] = 0


# In[48]:


# export not_enough_exps & sim_performance_datasets
tau_data_qs_markdown = tau_data_qs.copy()
for trial in not_enough_exps:
    qs, data = trial
    tau_data_qs_markdown.loc[qs, data] = f'{tau_data_qs_markdown.loc[qs, data]}' + '⚠️'

new_columns = []
for i, data in enumerate(tau_data_qs_markdown.columns):
    if data in sim_performance_datasets:
        new_columns.append(f'{tau_data_qs_markdown.columns[i]}' + '🤔')
    else:
        new_columns.append(f'{tau_data_qs_markdown.columns[i]}')

tau_data_qs_markdown.columns = new_columns


# In[49]:


tau_data_qs_description = '''# Usefulness of query strategies\n\nMean difference of the query strategy from Uniform\n`*\' and `**\' mean reject pair $t$-test with significance level $0.05$ and $0.01$ respectively.\n
- The `⚠️\' means not enough number of repeated experiments.
- The `🤔\' means hard to differentiate the performance of different query strategies.\n\n'''

with open('./README.md', 'a') as f:
    '# Benchmark of pool-based active learning\n\nMean(Standard Deviation) of Uniform (Random Sampling), 17 query strategies and Beam-Search Oracle (BSO) on 26 binary datasets.\n\n'
    f.write(tau_data_qs_description)
    f.write(tau_data_qs_markdown.to_markdown())


# ## Figure. Number of significant improvement of Query Strategy from Uniform
# 
# Count the number of (query strategy, *dataset*) that mean AUBC difference is greater than $0$ with $\alpha=5\%$.

# In[50]:


tau_data_qs_cnt_dataview = tau_data_qs_cnt.gt(0).sum()
tau_data_qs_cnt_qsview = tau_data_qs_cnt.gt(0).sum(axis=1)


# In[51]:


fig, (ax_data, ax_qs) = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
tau_data_qs_cnt_dataview.plot(kind='barh', ax=ax_data)
tau_data_qs_cnt_qsview.plot(kind='barh', ax=ax_qs)
ax_data.set_title('Dataset aspect')
ax_data.set_ylabel('')
ax_data.invert_yaxis()
ax_data.set_xlim(0, 17)
ax_qs.set_title('Query strategy aspect')
ax_qs.set_ylabel('')
ax_qs.set_xlim(0, 26)
ax_qs.invert_yaxis()
fig.savefig('n_QSgtRS.png', bbox_inches='tight')
plt.clf()


# ## Figure. Improvement of Query Strategies over Uniform on *Heart*
# 
# Compare the $\tau_{q, s, k}$, where 
# - $q \in$ {US, QBC, ALBL, Margin (InfoDiv), MCM, LAL, KCenter, EER, SPAL}.
# - $s = $ *Heart*

# In[52]:


useful_al = ['us', 'qbc', 'albl', 'margin', 'mcm', 'lal', 'kcenter', 'eer', 'spal']
data = 'heart'
useful_al_mcolor = {
    'us': 'salmon',
    'qbc': 'tab:orange',
    'albl': 'tab:pink',
    'kcenter': 'tab:green',
    'margin': 'tab:red',
    'mcm': 'tab:blue',
    'eer': 'brown',
    'spal': 'tab:cyan',
    'lal': 'magenta',
}


# In[53]:


tau_useful_heart_seed = tau_qs_data_seed['heart'].loc[:, useful_al]


# In[54]:


plt.rcParams["figure.figsize"] = (12, 8)
tau_useful_heart_seed.plot(marker='.', linestyle='--', alpha=0.5, ms=10)
plt.xlabel('Index of Seed')
plt.ylabel('Difference AUBC between QS from Uniform')
plt.axhline(y=0, linestyle='--', color='black')
plt.legend()
plt.savefig('scat-rank.png', bbox_inches='tight')
plt.clf()


# ## Table. Average Ranking of the Query Startegy
# 
# Apply Friedman test with $\alpha=0.05$ on $\bar{\tau}_{q, s}$.

# In[55]:


avg_rank_tau_qs_data = {}
rank_tau_qs_data = {}
for data in tau_qs_data_seed:
    cur_data = tau_qs_data_seed[data]
    cur_data = cur_data.loc[:, useful_al]
    cur_data = cur_data.dropna(axis=1, how='all')
    cur_data = cur_data.dropna()
    avg_rank_tau_qs_data[data] = cur_data.rank(axis=1, ascending=False).mean()
    rank_tau_qs_data[data] = cur_data.rank(axis=1, ascending=False)


# In[56]:


# Friedman test
for i, data in enumerate(rank_tau_qs_data):
    # Perform the Friedman test
    friedman_test, p_value = stats.friedmanchisquare(*[rank_tau_qs_data[data].values[:, i] for i in range(rank_tau_qs_data[data].values.shape[1])])
    
    if p_value < 0.05:
        print(f'{i}: {data} p-value is significant')


# In[57]:


avg_rank_tau_qs_data = pd.DataFrame(avg_rank_tau_qs_data)
avg_rank_tau_qs_data


# In[58]:


# pd.DataFrame to LaTeX
# largest three values in each column
# make them as bold, bold, underline
lbracebracket = f'{chr(123)}'
rbracebracket = f'{chr(125)}'
tbf = f'{chr(92)}textbf'
tit = f'{chr(92)}textit'
udl = f'{chr(92)}underline'


# In[59]:


# add rank
rank1_str = '\\textsuperscript{1}'
rank2_str = '\\textsuperscript{2}'
rank3_str = '\\textsuperscript{3}'
avg_rank_tau_qs_data_latex = avg_rank_tau_qs_data.round(2).copy()
for d in avg_rank_tau_qs_data.columns:
    bst_q = avg_rank_tau_qs_data.loc[:, d].nsmallest(3).index
    # export to LaTeX
    avg_rank_tau_qs_data_latex.loc[bst_q[0], d] = f'{tbf}{lbracebracket}{avg_rank_tau_qs_data_latex.loc[bst_q[0], d]}{rbracebracket}{rank1_str}'
    avg_rank_tau_qs_data_latex.loc[bst_q[1], d] = f'{tbf}{lbracebracket}{avg_rank_tau_qs_data_latex.loc[bst_q[1], d]}{rbracebracket}{rank2_str}'
    avg_rank_tau_qs_data_latex.loc[bst_q[2], d] = f'{tbf}{lbracebracket}{avg_rank_tau_qs_data_latex.loc[bst_q[2], d]}{rbracebracket}{rank3_str}'


# In[60]:


# add reason for undone experiments
avg_rank_tau_qs_data_latex = avg_rank_tau_qs_data_latex.fillna('too long')
avg_rank_tau_qs_data_latex.loc['spal', 'checkerboard'] = 'error'


# In[61]:


avg_rank_tau_qs_data_latex = avg_rank_tau_qs_data_latex.T
avg_rank_tau_qs_data_latex.columns.name = ''


# In[62]:


# update columns
avg_rank_tau_qs_data_latex_str = avg_rank_tau_qs_data_latex.to_latex(
    label='tab6:super',
    caption='Average Ranking of the Query Startegy',
    escape=False
)

avg_rank_tau_qs_data_latex_str = avg_rank_tau_qs_data_latex_str.replace('{table}', '{table*}')
with open('table6-super.tex', 'w') as f:
    f.write(avg_rank_tau_qs_data_latex_str)


# In[63]:


aubc_qs_data_seed_df = []
for data in aubc_qs_data_seed:
    aubc_q_d_s_ = aubc_qs_data_seed[data].copy()
    aubc_q_d_s_ = pd.DataFrame(aubc_q_d_s_)
    aubc_q_d_s_.columns = qs_list
    aubc_q_d_s_ = aubc_q_d_s_.stack()
    aubc_q_d_s_ = aubc_q_d_s_.reset_index()
    aubc_q_d_s_['data'] = data
    aubc_q_d_s_.columns = ['res_expno', 'qs', 'res_tst_score', 'data']
    aubc_q_d_s_ = aubc_q_d_s_[['data', 'res_expno', 'qs', 'res_tst_score']]
    aubc_qs_data_seed_df.append(aubc_q_d_s_)

aubc_qs_data_seed_df = pd.concat(aubc_qs_data_seed_df)


# In[64]:


aubc_qs_data_seed_view = pd.merge(left=aubc_qs_data_seed_df,
                                  right=xz2021_table2,
                                  how='left',
                                  left_on='data',
                                  right_index=True)


# In[65]:


aubc_RS_data_seed_view = aubc_qs_data_seed_view[aubc_qs_data_seed_view['qs']=='uniform']
aubc_US_data_seed_view = aubc_qs_data_seed_view[aubc_qs_data_seed_view['qs']=='margin']

tau_US_data_seed_view = []
for data in data_list:
    aubc_RS_seed_view = aubc_RS_data_seed_view[
        aubc_RS_data_seed_view['data']==data
    ].set_index('res_expno')
    aubc_US_seed_view = aubc_US_data_seed_view[
        aubc_US_data_seed_view['data']==data
    ].set_index('res_expno')
    tau_US_seed_view = aubc_US_seed_view['res_tst_score'] - aubc_RS_seed_view['res_tst_score']
    tau_US_seed_view = tau_US_seed_view.to_frame()
    tau_US_seed_view['IR'] = aubc_US_seed_view['IR']
    tau_US_seed_view['d'] = aubc_US_seed_view['d']
    tau_US_seed_view['n'] = aubc_US_seed_view['n']
    tau_US_seed_view['data'] = data
    tau_US_seed_view = tau_US_seed_view.reset_index(drop=True)
    tau_US_data_seed_view.append(tau_US_seed_view)

tau_US_data_seed_view = pd.concat(tau_US_data_seed_view)
tau_US_data_seed_view.columns = ['margin_improve'] + tau_US_data_seed_view.columns.to_list()[1:]


# In[66]:


tau_US_data_seed_view['margin_improve'] = tau_US_data_seed_view['margin_improve'].astype(float)


# In[67]:


cmap = plt.cm.jet  # define the colormap
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
cmapidx = np.linspace(0, len(cmaplist)-1, len(data_list)).astype(int)
cmapdict = {}
for i, d in zip(cmapidx, data_list):
    cmapdict[d] = plt.matplotlib.colors.rgb2hex(cmaplist[i])


# In[68]:


tau_US_data_seed_view['color'] = tau_US_data_seed_view['data'].map(cmapdict)


# In[69]:


from pandas.plotting import scatter_matrix

ax_tau_US_scat_mat = scatter_matrix(tau_US_data_seed_view,
                                    alpha=0.5,
                                    diagonal='kde',
                                    c=tau_US_data_seed_view['color'],
                                    s=100,
                                    figsize=(12, 8))
ax_tau_US_corr = tau_US_data_seed_view.corr().to_numpy()
for i, j in zip(*plt.np.triu_indices_from(ax_tau_US_scat_mat, k=1)):
    ax_tau_US_scat_mat[i, j].annotate("r=%.3f" %ax_tau_US_corr[i, j],
                                      (0.8, 0.8),
                                      xycoords='axes fraction',
                                      ha='center',
                                      va='center')
    ax_tau_US_scat_mat[i, j].xaxis.set_visible(True)
    if j == 1:
        ax_tau_US_scat_mat[i, j].yaxis.set_visible(True)

for i in range(4):
    for j in range(4):
        if i > 0:
            ax_tau_US_scat_mat[i,j].set_visible(False)
        if j == 0:
            ax_tau_US_scat_mat[i,j].set_visible(False)

plt.savefig('scatmat.png', bbox_inches='tight')
plt.clf()


# In[ ]:




