# test_embeddings.py
from joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn import linear_model

mem = Memory("./mycache")
@mem.cache
def get_data(path="cifar10_svmstyle.txt"):
    data = load_svmlight_file(path)
    return data[0], data[1]

X_cifar10, y_cifar10 = get_data()
# split cifar10 into train and test
X_cifar10_train = X_cifar10[:50000]
y_cifar10_train = y_cifar10[:50000]
X_cifar10_test = X_cifar10[50000:]
y_cifar10_test = y_cifar10[50000:]
print(X_cifar10.shape, y_cifar10.shape)
logistic = linear_model.LogisticRegression(solver='liblinear')
score = logistic.fit(X_cifar10_train, y_cifar10_train).score(X_cifar10_test, y_cifar10_test)
print(f'CIFAR10 LIBLINEAR with C=1, score {score}')

X_imdb, y_imdb = get_data("imdb_svmstyle.txt")
X_imdb_train = X_imdb[:25000]
y_imdb_train = y_imdb[:25000]
X_imdb_test = X_imdb[25000:]
y_imdb_test = y_imdb[25000:]
print(X_imdb.shape, y_imdb.shape)
logistic = linear_model.LogisticRegression(solver='liblinear')
score = logistic.fit(X_imdb_train, y_imdb_train).score(X_imdb_test, y_imdb_test)
print(f'IMDB LIBLINEAR with C=1, score {score}')
