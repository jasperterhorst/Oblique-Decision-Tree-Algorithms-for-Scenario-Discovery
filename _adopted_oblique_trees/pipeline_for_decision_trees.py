# .............Pipeline for calling the Oblique Decision Trees using the Scikit-Learns Bagging Classifier..............


# Importing all the oblique decision trees
from _adopted_oblique_trees.WODT import *
from _adopted_oblique_trees.HouseHolder_CART import *
from _adopted_oblique_trees.RandCART import *
from _adopted_oblique_trees.CO2 import *
from _adopted_oblique_trees.Oblique_Classifier_1 import *
from _adopted_oblique_trees.segmentor import *
from _adopted_oblique_trees.split_criteria import *

# Importing all the packages
from sklearn.tree import DecisionTreeClassifier


def make_estimator(method='wodt', max_depth=5, n_estimators=10):
    if method == 'wodt':
        return WeightedObliqueDecisionTreeClassifier(max_depth=max_depth)
    elif method == 'oc1':
        return ObliqueClassifier1(max_depth=max_depth)
    elif method == 'stdt':
        return DecisionTreeClassifier(max_depth=max_depth)
    elif method == 'hhcart_a':
        return HHCartAClassifier(mse, CARTSegmentor(), max_depth=max_depth)
    elif method == 'hhcart_d':
        return HHCartDClassifier(mse, CARTSegmentor(), max_depth=max_depth)
    elif method == 'randcart':
        return RandCARTClassifier(mse, MeanSegmentor(), max_depth=max_depth)
    elif method == 'co2':
        return CO2Classifier(mse, CARTSegmentor(), max_depth=max_depth)
    else:
        ValueError('Unknown model!')
