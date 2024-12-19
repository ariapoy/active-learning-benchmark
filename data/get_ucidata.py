from ucimlrepo import fetch_ucirepo 
import numpy as np
from sklearn.preprocessing import LabelEncoder

id_data_dict = {
    # toy
    53: 'iris',
    149: 'vehicle',
    109: 'wine',
    # climate
    146: 'satellite',
    # business
    186: 'winequality',
    # health
    579: 'myocardial',
    # biology
    602: 'bean',
    # social
    697: 'academic',
}

def get_data(id):
    data = fetch_ucirepo(id=id)
    # data (as pandas dataframes) 
    X = data.data.features 
    y = data.data.targets 
    # Save embeddings in LibSVM format
    with open(f'{id_data_dict[id]}-svmstyle.txt', 'w') as f:
        X = X.values
        y = y.values
        y = y.reshape(-1, )
        y = LabelEncoder().fit_transform(y)
        for idx in range(X.shape[0]):
            features = X[idx, :]
            target = y[idx]
            # skip nan or non-nunmeric values
            features = ' '.join([f"{i+1}:{value}" for i, value in enumerate(features) if np.isfinite(value)])
            f.write(f"{target} {features}\n")
    return X, y

for id in id_data_dict.keys():
    X, y = get_data(id)
    print(id_data_dict[id])
    print(X.shape, len(np.unique(y)))
    print(np.unique(y, return_counts=True))