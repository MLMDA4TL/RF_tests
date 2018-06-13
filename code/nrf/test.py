from Neural_Decision_Tree import ndt
from sklearn.tree import DecisionTreeClassifier,export_graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from keras.regularizers import l1
sys.path.append("../data_mngmt")
from data import load_letter

X_s,X_t_train,X_t_test,y_s,y_t_train,y_t_test, = load_letter()

clf = DecisionTreeClassifier(max_depth=6)
clf = clf.fit(X_s, y_s)

a = ndt(D=X_s.shape[1],gammas=[1,10,1],tree_id=0)
a.compute_matrices_and_biases(clf)
a.to_keras(dropouts=[0.1,0.1,0.1],kernel_regularizer=[l1(1e-5),l1(1e-5),l1(1e-5)])


z = a.get_activations(X_s[:1,:])
#-----
print "Target score before training"
print a.score(X_t_test,y_t_test), clf.score(X_t_test,y_t_test)
print "Source score before training"
print a.score(X_s,y_s), clf.score(X_s,y_s)
print "train on source"
#-----
errors = a.fit(X_s,y_s,epochs=100)
print "Target score after training"
print a.score(X_t_test,y_t_test), clf.score(X_t_test,y_t_test)
print "Source score after training"
print a.score(X_s,y_s), clf.score(X_s,y_s)
#----
print "get weights"
a.get_weights_from_NN()
differences = a.compute_weights_differences()
a.plot_old_new_network()
a.plot_differences()
a.plot_W_nn_quantiles()

#-----
print "refine on target"
errors = a.fit(X_t_train,y_t_train,epochs=100)
#-----
print "Target scores after training"
print a.score(X_t_test,y_t_test)
print "Source score before training"
print a.score(X_s,y_s)
#----
print "get weights"
a.get_weights_from_NN()
differences = a.compute_weights_differences()
a.plot_old_new_network()
a.plot_differences()
a.plot_W_nn_quantiles()
