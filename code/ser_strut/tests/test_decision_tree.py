import sys
import graphviz
import pickle
import copy
import matplotlib.pyplot as plt
sys.path.append('../')
from STRUT import *

D = 3
dataset_length = 100

X = np.random.randn(dataset_length, D) * 0.1
Y = np.ones(dataset_length)
Y[0:dataset_length // 2] *= 0

# Train a Tree
clf = tree.DecisionTreeClassifier(max_depth=None)
clf = clf.fit(X, Y)
out_1 = clf.predict(X)

# put leafs values at zero
dt = clf.tree_
dt.value[dt.feature == -2] = 0
dt.n_node_samples[dt.feature == -2] = 0
dt.weighted_n_node_samples[dt.feature == -2] = 0
out_2 = clf.predict_proba(X)
