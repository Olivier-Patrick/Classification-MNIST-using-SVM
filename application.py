from __future__ import division, print_function
import numpy as np
from sklearn import datasets

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from utils import *
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from supervised_learning import SVMClassifier


def y_binaire(y, i):
    """ Stratégie tous contre un , on construit de nouvelle étiquette d'une part on a
    la classe comportant le chiffre i et d'autre part la classe ne comportant pas le chiffre i

    :param y: données de label
    :param i: le chiffre de la classe_0
    :return: y_ les données de label ré-étiquetés
    """
    if (int(i) >= 0) & (int(i) <= 9):
        y_= ( y == int(i))
    else:
        print("Désolé les labels sont entre 0 et 9")
    return y_

def y_multi_label(y_,**list_class):
    """
    :param y_: données de label
    :param list_class: est une liste de liste où chaque liste comporte les éléménts de la même classe, le nombre d'élément
           de list_class est le nombre de class formé.
    :return: y_ les données de label ré-étiquetés
    """

    for k in range(y_.shape[0]):
        for list in list_class:
            if y_[k] in list:
                y_[k] = list[0]
    return y_

def main():
    mnist = datasets.load_digits()
    X = mnist['data']/225.
    y = mnist['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y_binaire(y, 0), test_size=0.33)

    parameters = {'c': [0.1,1],
              'gamma': [1,0.1],
              "kernel" :  ['rbf','poly']}
    svm = SVMClassifier()
    clf = GridSearchCV(svm, parameters,cv=3, verbose=3)
    clf.fit(X_train,y_train)

    print(35*'\n. ' + 'MEILLEUR PARAMÈTRE' + 35*' .')
    print("best parametre", clf.best_params_)




if __name__ == "__main__":
    main()