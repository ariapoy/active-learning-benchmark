# check dataset properties
from sklearn.datasets import load_svmlight_file
import os
import glob
import numpy as np
import pandas as pd

path = 'dataset_used_in_ALSurvey'
extension = 'txt'
data_set_list = glob.glob('{}/*.{}'.format(path, extension))

data_property = {
    'dataset': [],
    'num_instance': [],
    'num_features': [],
    'num_classes': [],
    'IR': [],
    'class_distribution': []
}

for data_set in data_set_list:
    data = load_svmlight_file("{0}".format(data_set))
    name = data_set.split('/')[1].split('.')[0].split('-')[0]
    X, y = data[0], data[1]
    X = np.asarray(X.todense())
    class_dist = np.unique(y, return_counts=True)
    imbalance_ratio = class_dist[1].max()/class_dist[1].min()
    # print(f"X.shape: {X.shape}")
    # print(f"class num: {len(class_dist[0])}")
    # print(f"class distribution: {class_dist}")
    # print(f"IR: {imbalance_ratio}")

    data_property['dataset'].append(name)
    data_property['num_instance'].append(X.shape[0])
    data_property['num_features'].append(X.shape[1])
    data_property['num_classes'].append(len(class_dist[0]))
    data_property['IR'].append(class_dist[1].max()/class_dist[1].min())
    data_property['class_distribution'].append(str({c: n for c, n in zip(class_dist[0], class_dist[1])}))

data_property = pd.DataFrame(data_property)
data_property.to_csv('datasets_property.csv')