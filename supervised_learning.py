from __future__ import division, print_function
import cvxopt
from utils import *
from sklearn.base import BaseEstimator, ClassifierMixin
# Cachons la sortie de cvxopt
cvxopt.solvers.options['show_progress'] = False

class SVMClassifier(BaseEstimator, ClassifierMixin):
   """
   Une classe de classifieur SVM (Support Vector Machine)

   Les attributs:
   - gamma : l'hyper-paramètre (réel)
   - kernel: le noyau utilisé (string: rbf, poly, lin)
   - kernel_: la fonction réelle du noyau
   - x : les données sur lesquelles le SVM est formé (vecteurs de support)
   - y : les cibles des données d'entraînement
   - coef_ : coefficients des vecteurs de support
   - intercept_ : terme d'interception
   """

   def __init__(self, gamma:float=1.0, kernel:str=None, c:float=1.0, power:float=2, sigma:float=1.0):
      self.gamma=gamma
      self.c=c
      self.power=power
      self.sigma=sigma
      if (kernel is None):
         self.kernel='rbf'
      else:
         self.kernel=kernel

      params=dict()
      if (kernel=='poly'):
         params['c']=c
         params['power']=power
      elif (kernel=='rbf'):
         params['sigma']=sigma

      self.kernel_=SVMClassifier.__set_kernel(self.kernel,**params)

      self.x=None
      self.y=None
      self.coef_=None
      self.intercept_=None

   def get_params(self, deep=True):
       """
           La fonctionnalité get_params fournit les paramètres de la classe SVMClassifier.
            Ceux-ci excluent les paramètres du modèle.
       """
       return {"c": self.c, "power": self.power, "gamma": self.gamma,
               "kernel": self.kernel, "sigma": self.sigma}

   def set_params(self, **parameters):
       """ Définissez les paramètres de la classe. Remarque importante: cela devrait faire
            tout ce qui est fait aux paramètres pertinents dans __init__ comme
            GridSearchCV de sklearn l'utilise à la place de init.

           Plus info:  https://scikit-learn.org/stable/developers/develop.html
       """
       for parameter, value in parameters.items():
           # setattr devrait faire l'affaire pour gamma, c, d, sigma et kernel
           setattr(self, parameter, value)
       # maintenant aussi mettons à jour le noyau réel
       params = dict()
       if self.kernel == 'poly':
           params['c'] = self.c
           params['power'] = self.power
       elif self.kernel == 'rbf':
           params['sigma'] = self.sigma
       self.kernel_ = SVMClassifier.__set_kernel(self.kernel, **params)

       return self

   def set_attributes(self, **parameters):
       """ Définissez manuellement les attributs du modèle. Cela devrait généralement
            ne pas être fait, sauf lors du test d'un comportement spécifique, ou
            créer un modèle moyenné.
            Les paramètres sont fournis sous forme de dictionnaire.

               - 'intercept_' : réel intercept
               - 'coef_'      :  tableau de coefficients réel
               - 'support_'   : tableau de vecteurs de support, dans le même ordre triés
                                 comme les coefficients
       """
       # pas la manière la plus efficace de le faire ... mais suffisante pour le moment
       for param, value in parameters.items():
           if param == 'intercept_':
               self.intercept_ = value
           elif param == 'coef_':
               self.coef_ = value
           elif param == 'c':
               self.c = value
           elif param == 'power':
               self.power = value
           elif param == 'support_':
               self.x = value

   @staticmethod
   def __set_kernel(name: str, **params):
       """
           Fonction statique interne pour définir la fonction du noyau.
             NOTE: Le deuxième «vecteur» xj sera celui qui généralement
                   contient un tableau de vecteurs possibles, tandis que xi doit être un seul
                   vecteur. Par conséquent, le produit scalaire numpy nécessite xj pour
                   être transposée.
             Le noyau renvoie un scalaire ou un nd-array numpy de
             rang 1 (c'est-à-dire un vecteur), s'il renvoie autre chose, le résultat
             est faux si xi est un tableau.
       """

       def linear(xi, xj):
           """
              v*v=scal (dot-product OK)
              v*m=v    (dot-product OK)
              m*m=m    (matmul pour 2Dx2D, ok avec le produit scalaire)
           """
           return np.dot(xi, xj.T)

       def poly(xi, xj, c=params.get('c', 1.0) ,power=params.get('d', 2)):
           """
               Polynomial kernel ={1+ (xi*xj^T)/c }^power
               Paramètre:
                   - c: constante de mise à l'échelle, DEFAULT=1.0
                   - power: puissance polynomiale, DEFAULT=2
                   - xi et xj sont numpy nd-arrays
               (ref: https://en.wikipedia.org/wiki/Least-squares_support-vector_machine )
               fonctionne de la même manière que linéaire
           """
           return ((np.dot(xi, xj.T)) / c + 1) ** power

       def rbf(xi, xj, sigma=params.get('sigma', 1.0)):
           """
           Fonction de base radiale kernel= exp(- ||xj-xi||² / (2*sigma²))
           Dans cette formulation, le rbf est également connu sous le nom de noyau gaussien de variance sigma²
           Comme la distance euclidienne est strictement positive, les résultats de ce noyau
           sont dans l'interval [0..1] (x € [+infty..0])
           Paramètres:
               - sigma: constante de mise à l'échelle, DEFAULT=1.0
               - xi et xj sont numpy nd-arrays
           (cf: https://en.wikipedia.org/wiki/Least-squares_support-vector_machine )
            combinations possibles of xi and xj:
               vect & vect   -> scalar
               vect & array  -> vect
               array & array -> array => celui-ci nécessite une distance de paire ...
                                   ce qui ne peut pas être fait avec matmul et dot
               Les vecteurs sont les lignes du arrays (Arr[0,:]=first vect)
               La distance au carré entre vectors= sqr(sqrt( sum_i(vi-wi)² ))
               --> sqr & sqrt
               --> vous pouvez utiliser un opérateur de produit scalaire pour les vecteurs ... mais ceci
                semble échouer pour nd-arrays.
           Pour les vecteurs:
               ||x-y||²=sum_i(x_i-y_i)²=sum_i(x²_i+y²_i-2x_iy_i)
               --> tous les produits entre vecteurs peuvent être réalisés via np.dot: prend les carrés & sommes
           Pour les vecteurs x et array de vecteurs y:
               --> x²_i : ce sont des vecteurs: le point donne un scalaire
               --> y²_i :cela devrait être une liste de scalaires, un par vecteur.
                           => np.dot donne un 2d array
                           => ainsi   1) square manuellement (le carré de chaque element)
                                   2) sum sur chaque ligne (axis=1...mais seulement au cas où nous
                                                           avons un 2D array)
               --> x_iy_i : cela devrait aussi être une liste de scalaires. np.dot fait l'affaire,
                           et donne même le même résultat si matrice et vecteur sont échangés
            pour tableau de vecteurs x et tableau de vecteurs y:
               --> soit boucle sur les vecteurs de x, et pour chacun faire ce qui précède
               --> ou utilisons cdist qui calcule la distance par paire et utilisons-la dans exp
           """
           from scipy.spatial.distance import cdist

           if (xi.ndim == 2 and xi.ndim == xj.ndim):  # les deux sont des matrices 2D
               return np.exp(-(cdist(xi, xj, metric='sqeuclidean')) / (2 * (sigma ** 2)))
           elif ((xi.ndim < 2) and (xj.ndim < 3)):
               ax = len(xj.shape) - 1  # compensate for python zero-base
               return np.exp(-(np.dot(xi, xi) + (xj ** 2).sum(axis=ax)
                               - 2 * np.dot(xi, xj.T)) / (2 * (sigma ** 2)))
           else:
               message = "Le noyau rbf n\'est pas adapté aux tableaux avec rang >2"
               raise Exception(message)

       kernels = {'linear': linear, 'poly': poly, 'rbf': rbf}
       if kernels.get(name) is not None:
           return kernels[name]
       else:  # unknown kernel: crash and burn?
           message = "Kernel " + name + " n\'est pas mis en œuvre. Veuillez choisir parmi: "
           message += str(list(kernels.keys())).strip('[]')
           raise KeyError(message)

   def fit(self, X, y):

        n_samples, n_features = np.shape(X)

        # Parametrons gamma = 1/n_features par défaut
        if not self.gamma:
           self.gamma = 1 / n_features

        # Initialisons la methode du noyeau avec des parametres
        #self.kernel = self.kernel(power=self.power,gamma=self.gamma,coef_=self.coef_)

        # Calculons la matrice du noyeau
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.kernel_(X[i], X[j])

        # Définition du problème d'optimisation quadratique
        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix, tc='d')
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y.astype(np.double), (1, n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')

        if not self.c:
           G = cvxopt.matrix(np.identity(n_samples) * -1)
           h = cvxopt.matrix(np.zeros(n_samples))
        else:
           G_max = np.identity(n_samples) * -1
           G_min = np.identity(n_samples)
           G = cvxopt.matrix(np.vstack((G_max, G_min)))
           h_max = cvxopt.matrix(np.zeros(n_samples))
           h_min = cvxopt.matrix(np.ones(n_samples) * self.c)
           h = cvxopt.matrix(np.vstack((h_max, h_min)))

        # Résolution du problème d'optimisation quadratique à l'aide de cvxopt
        minimization = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        lagr_mult = np.ravel(minimization['x'])

        # Extraction des vecteurs de support
        # Obtention des indexes du lagranges multiplicateurs non nulles.
        idx = lagr_mult > 1e-7
        # Obtention du lagranges multiplicateurs correspondant
        self.lagr_multipliers = lagr_mult[idx]

        # Obtention des X qui serviront de vecteurs de support
        self.support_vectors = X[idx]

        # Obtention des étiquettes correspondantes
        self.support_vector_labels = y[idx]

        # Calcul de l'interception avec le premier vecteur de support
        self.intercept_ = self.support_vector_labels[0]
        for i in range(len(self.lagr_multipliers)):
            self.intercept_ -= self.lagr_multipliers[i] * self.support_vector_labels[
              i] * self.kernel_(self.support_vectors[i], self.support_vectors[0])

   def predict(self, X):
       y_pred = []
       # Parcourons X et faisons des prédictions
       for sample in X:
           prediction = 0
           # Détermination de l'étiquette des X par les vecteurs de support
           for i in range(len(self.lagr_multipliers)):
               prediction += self.lagr_multipliers[i] * self.support_vector_labels[
                   i] * self.kernel_(self.support_vectors[i], sample)
           prediction += self.intercept_
           y_pred.append(np.sign(prediction))
       return np.array(y_pred)

   def get_acc(preds, labels, print_conf_matrix=False):
       act_pos = sum(labels == 1)
       act_neg = len(labels) - act_pos
       ## prediction positive
       pred_pos = sum(1 for i in range(len(preds)) if (preds[i] >= 0.5))

       ## prediction negative
       true_pos = sum(1 for i in range(len(preds)) if (preds[i] >= 0.5) & (labels[i] == 1))

       ## faux positive
       false_pos = pred_pos - true_pos

       ## faux negative
       false_neg = act_pos - true_pos

       ## vrai negative
       true_neg = act_neg - false_pos

       ## tp/(tp+fp) pourcentages des predictions positives correctement classifiés
       precision = true_pos / pred_pos

       ## tp/(tp+fn) pourcentage des positives correctement classifiés
       recall = true_pos / act_pos

       f_score = 2 * precision * recall / (precision + recall)

       if print_conf_matrix:
           print('\nconfusion matrix')
           print('----------------')
           print('tn:{:6d} fp:{:6d}'.format(true_neg, false_pos))
           print('fn:{:6d} tp:{:6d}'.format(false_neg, true_pos))

       return (f_score)