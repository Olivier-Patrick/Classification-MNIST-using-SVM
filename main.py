
from sklearn import datasets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_openml

#X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
#X = X / 255.
#y = y.astype('int64')

mnist = datasets.load_digits()
X = mnist['data']
y = mnist['target']

def visualise_chiffre(X,y,i):
    plt.imshow(X[i].reshape(8, 8), cmap = matplotlib.cm.binary, interpolation = "nearest")
    plt.title('label:' + str(y[i]))
    plt.axis("on")
    plt.show()

def division_data(X,y,test_size = 0.25):
    X_train, X_test, y_train, y_test = X[:round(X.shape[0]*test_size)], X[round(X.shape[0]*test_size):], y[:round(X.shape[0]*test_size)], y[round(X.shape[0]*test_size):]
    shuffle_index = np.random.permutation(round(X.shape[0]*test_size))
    X_train, y_train,X_test, y_test = X_train[shuffle_index], y_train[shuffle_index],X_test[shuffle_index], y_test[shuffle_index]

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = division_data(X,y,0.20)


def classifieur_binaire(X_train,X_test,y_train,y_test,i):

    if (int(i) >= 0) & (int(i) <=9):
       y_train_binaire = (y_train == int(i))
       y_test_binaire = (y_test == int(i))

       svc = SVC()
       svc.fit(X_train, y_train_binaire)
       y_pred_binaire = svc.predict(X_test)
       print(confusion_matrix(y_test_binaire,y_pred_binaire))

    else: print("Désolé les labels sont entre 0 et 9")


def classifieur_multi_labels(X_train, X_test, y_train, y_test, list_class=[]):

    for k in range(y_train.shape[0]):
        for list in list_class:
            if y_train[k] in list:
                y_train[k] = list[0]

    for k in range(y_test.shape[0]):
        for list in list_class:
            if y_test[k] in list:
                y_test[k] = list[0]

    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}

    grid = GridSearchCV(SVC(), param_grid, cv=5, verbose=3)

    grid.fit(X_train, y_train)

    grid_predictions = grid.predict(X_test)

    print(100 * '-')
    print(grid.best_params_)

    print(100 * '-')
    print(classification_report(y_test, grid_predictions))


classifieur_binaire(X_train,X_test,y_train,y_test,2)

print(100*'-')
classifieur_multi_labels(X_train,X_test,y_train,y_test,[[1,2,5],[4,6]])
