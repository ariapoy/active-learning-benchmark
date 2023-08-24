import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

query_models = ['LR', 'RBFSVM', 'RandomForest']
task_models = ['LR', 'RBFSVM', 'RandomForest']
# datasets = ['sonar', 'ionosphere', 'gcloudb', 'checkerboard', 'banana', 'haberman', 'tic', 'appendicitis', 'breast', 'twonorm']
datasets = ["appendicitis", "sonar", "parkinsons", "ex8b", "heart", "haberman", "ionosphere", "clean1", "breast", "wdbc", "australian", "diabetes", "mammographic", "ex8a", "tic", "german", "splice", "gcloudb", "gcloudub", "checkerboard", "spambase", "banana", "phoneme", "ringnorm", "twonorm", "phishing"]
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

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    #cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    #cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    cbar = None
    cbar = None

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels, fontsize=18)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels, fontsize=18)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #         rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

for data in aubc_avg_3x3:
    print(data)
    print(aubc_avg_3x3[data])
    bst_q_model = aubc_avg_3x3[data].astype(float).max(axis=1).idxmax()
    bst_t_model = aubc_avg_3x3[data].astype(float).max().idxmax()
    assert aubc_avg_3x3[data].loc[bst_q_model, bst_t_model] == aubc_avg_3x3[data].astype(float).max().max()
    print(f'Query model: {bst_q_model: <12} X Task model: {bst_t_model}')
    fig, ax = plt.subplots(figsize=(12,8))
    plt.rcParams.update({'font.size': 24})
    values = aubc_avg_3x3[data].values.astype(float)
    idx_name = ['LR(C=1)', 'SVM(RBF)', 'RF'] # aubc_avg_3x3[data].index
    col_name = ['LR(C=1)', 'SVM(RBF)', 'RF'] # aubc_avg_3x3[data].columns
    im, _ = heatmap(values, idx_name, col_name, ax=ax, cmap="YlGn")
    texts = annotate_heatmap(im, valfmt="{x:.2%}")

    #ax.set_xticks(np.arange(len(idx_name)), labels=idx_name, fontsize=18)
    #ax.xaxis.tick_top()
    #ax.set_yticks(np.arange(len(col_name)), labels=col_name, fontsize=18)
    #for i in range(len(idx_name)):
    #    for j in range(len(col_name)):
    #        text = ax.text(j, i, f'{values[i, j]: .2%}', ha="center", va="center")

    export_name = f'diffmodels-{data}'
    fig.tight_layout()
    # plt.savefig(f'../../Poy2023a/images/{export_name}.eps', bbox_inches='tight', format='eps', dpi=200)
    plt.savefig(f'images/{export_name}.png', bbox_inches='tight')
    plt.clf()
