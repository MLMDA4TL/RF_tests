# -*- coding: utf-8 -*-
"""
@author: sergio
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import sys
sys.path.append("../utils")
from common_functions import find_parent,leaves_id,get_list_split_phi_forest,get_list_split_phi,get_parents_nodes_leaves_dic
from collections import OrderedDict 

from Neural_Decision_Tree import ndt
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras import optimizers
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from keras.regularizers import l2,l1


class nrf_fully_connected(ndt):
  def __init__(self, D, gammas=[10,1,1,1],kernel_regularizer=[l1(0),l1(0),l1(0)],sigma=0):
    ndt.__init__(self, D, gammas,kernel_regularizer,sigma)

  def compute_matrices_and_biases(self, random_forest):
    self.rf = random_forest
    self.classes = self.rf.classes_
    self.ndts = []
    # Compute weights and biases of individual trees
    for i,dtree in enumerate(random_forest.estimators_):
      self.ndts.append(ndt(self.D, self.gammas, i))
      self.ndts[i].compute_matrices_and_biases(dtree)
    # Concatenate input to nodes W
    self.W_in_nodes = pd.concat([t.W_in_nodes for t in self.ndts],axis=1)
    # Concatenate input to nodes b
    self.b_nodes = pd.concat([t.b_nodes for t in self.ndts],axis=0)
    # Concatenate nodes to leaves W
    self.W_nodes_leaves = pd.concat([t.W_nodes_leaves for t in self.ndts],axis=1)
    self.W_nodes_leaves = self.W_nodes_leaves.fillna(0)
    # Concatenate nodes to leaves b
    self.b_leaves = pd.concat([t.b_leaves for t in self.ndts],axis=0)
    # Concatenate leaves to out W
    self.W_leaves_out = pd.concat([t.W_leaves_out for t in self.ndts],axis=0)
    # Concatenate leaves to out b
    self.b_class = pd.concat([t.b_class for t in self.ndts],axis=0)
    self.b_class = self.b_class.groupby(self.b_class.index).mean()
    # Set other parameters
    self.N = self.b_nodes.size
    self.L = self.b_leaves.size
    self.C = self.b_class.size

class nrf_independent_ndt(ndt):
  def __init__(self, D, gammas=[10,1,1,1]):
    ndt.__init__(self, D, gammas)

  def compute_weights_differences(self, random_forest):
    self.rf = random_forest
    self.classes = self.rf.classes_
    self.ndts = []
    for i,dtree in enumerate(random_forest.estimators_):
      self.ndts.append(ndt(self.D, self.gammas, i))
      self.ndts[i].compute_matrices_and_biases(dtree)
    # Define averaging layer
    self.W_outputs_to_output = pd.DataFrame(np.concatenate([np.eye(self.C)*self.gammas[-1] for tree in self.ndts], axis=0),
                                            index=[str(t.tree_id)+"_"+str(c) for t in self.ndts for t_class in t.classes],
                                            columns=self.classes)
    self.b_class = pd.DataFrame(np.zeros(self.C),
                                index = self.classes,
                                columns = ["CLASS_BIASES"])
  def to_keras(self):
    # Compute keras models for other trees
    for tree in self.ndts:
      tree.to_keras()
    # Define the averaging model
    self.concatenation_output_layers = Concatenate([t.output_layer for t in self.ndts])
    self.dropouts_output_layers = Dropout(dropouts[2])(self.concatenation_output_layers)
    self.output_layer = Dense(self.C, activation='softmax')(self.dropouts_output_layers)
    self.model = Model(input=[t.input_layer for t in self.ndts], output=self.output_layer)
    self.model.compile(loss=loss, optimizer=self.sgd)
    self.model.layers[8].set_weights(weights=[W_outputs_to_output,
                                              self.b_class])

  def get_activations(self,
                      X,
                      y=None):
    output_a = pd.DataFrame(self.model.predict(X),
                            columns=self.b_class.columns)
    return [t.get_activations(X,y) for t in self.ndts]+[[output_a]]

  def get_weights_from_NN(self):
    for tree in self.ndts:
      tree.get_weights_from_NN()
    w_9 = self.model.layers[9].get_weights()
    self.W_outputs_to_output_nn = pd.DataFrame(w_9[0],
                                               index=[str(t.tree_id)+"_"+str(c) for t in self.ndts for t_class in t.classes],
                                               columns=self.classes)
    self.b_class_nn = pd.DataFrame(w_9[1],
                                   index = self.classes,
                                   columns = ["CLASS_BIASES"])

  def compute_weights_differences(self):
    self.get_weights_from_NN()
    diff_W_outputs_to_output_nn = self.W_outputs_to_output - self.W_outputs_to_output_nn
    diff_b_class_nn = self.b_class_nn - self.b_class
    return [t.compute_weights_differences() for t in self.ndts] + [[diff_W_outputs_to_output_nn,diff_b_class_nn]]

  def print_tree_weights(self):
    for t in self.ndts:
      t.print_tree_weights()
    print("W: outputs -> output")
    print(self.W_outputs_to_output)
    print("b: output")
    print(self.b_class)

  def print_nn_weights(self):
    for t in self.ndts:
      t.print_nn_weights()
    print("W: outputs -> output")
    print(self.W_outputs_to_output_nn)
    print("b: output")
    print(self.b_class_nn)    

if __name__ == "__main__":
  np.random.seed(0)
  import matplotlib.pyplot as plt
  dataset_length = 1000
  D = 2
  X = np.random.randn(dataset_length,D)*0.1
  X[0:dataset_length//2,0] += 0.1
  X[0:dataset_length//2,0] += 0.2
  Y = np.ones(dataset_length)
  Y[0:dataset_length//2] *= 0

  X_test = np.random.randn(dataset_length,D)*0.1
  X_test[0:dataset_length//2,0] += 0.1
  X_test[0:dataset_length//2,0] += 0.2
  Y_test = np.ones(dataset_length)
  Y_test[0:dataset_length//2] *= 0
  # Train a Tree
  rf = RandomForestClassifier(max_depth=5,n_estimators=10)
  rf.fit(X, Y)

  a = nrf_fully_connected(D=2,gammas=[10,1,1],sigma=0.)
  a.compute_matrices_and_biases(rf)
  l1_coef = -4
  a.to_keras(kernel_regularizer=[l1(10**l1_coef),l1(10**l1_coef),l1(10**l1_coef)],
             dropouts = [0.1,0.5,0.5])
  print "scores before training"
  print a.score(X_test,Y_test)
  print a.score(X,Y)
  errors = a.fit(X,Y,epochs=100)
  plt.plot(errors)
  plt.show()
  print "scores after training"
  print a.score(X_test, Y_test)
  print a.score(X,Y)
  print "scores forest"
  print a.rf.score(X_test, Y_test)
  print a.rf.score(X,Y)
  a.get_weights_from_NN()

  #print "Tree weights"
  #a.print_tree_weights()
  #print "NN weights"
  #a.print_nn_weights()
  #print "activations"
  #print a.get_activations(X)
  differences = a.compute_weights_differences()
  a.plot_old_new_network()
  a.plot_differences()
  a.plot_W_nn_quantiles()


