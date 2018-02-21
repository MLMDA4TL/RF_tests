import numpy as np
import sklearn.tree as skl_tree
from sklearn.datasets import load_breast_cancer


data = load_breast_cancer()

DT = skl_tree.DecisionTreeClassifier()
DT.fit(data['data'], data['target'])

def tree_merge():


    return
