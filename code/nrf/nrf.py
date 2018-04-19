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


from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras import optimizers
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score

BIAS = 1
DIM = 0
class dt2nn:
  def __init__(self,decision_tree,D,gammas=[1000,1000,1000]):
    self.gammas = gammas
    self.splits = pd.DataFrame(get_list_split_phi(decision_tree)).T
    self.leaves = get_parents_nodes_leaves_dic(decision_tree)
    self.D = D
    self.N = self.splits.shape[0]
    self.L = len(self.leaves)
    # Fill Input -> Nodes layer matrix
    self.W_in_nodes = pd.DataFrame(np.zeros((self.D,self.N)),
                                   index = list(range(self.D)),
                                   columns = self.splits.index)
    for node, dim in self.splits[DIM].iteritems():
      self.W_in_nodes.loc[dim,node] = 1.*self.gammas[0]
    # Fill Input -> Nodes layer biases
    self.b_nodes = - self.splits[BIAS]*self.gammas[0]
    # Fill Nodes -> Leaves layer matrix
    self.W_nodes_leaves = pd.DataFrame(np.zeros((self.N,self.L)),
                                      index = self.splits.index,
                                      columns = self.leaves.keys())
    for leave,node_sides in self.leaves.iteritems():
      for node,r_l in node_sides:
        self.W_nodes_leaves.loc[node,leave] = r_l*self.gammas[1]
    # Fill Nodes -> Leaves layer biases       
    b_leaves = {k:-len(x)+0.5 for k,x in self.leaves.iteritems()}
    self.b_leaves = pd.DataFrame(b_leaves.values(), b_leaves.keys())
    self.b_leaves *= self.gammas[1]
    # Fill Leaves -> class matrix
    self.classes = decision_tree.classes_
    self.C = len(self.classes)
    class_counts_per_leave = decision_tree.tree_.value[self.leaves.keys()]
    class_counts_per_leave = class_counts_per_leave.reshape(self.L,self.C)
    self.W_leaves_out = pd.DataFrame(class_counts_per_leave,
                                     index = self.leaves.keys(),
                                     columns = self.classes)
    self.W_leaves_out = (self.W_leaves_out.T * 1./ self.W_leaves_out.T.sum()).T
    self.W_leaves_out *= self.gammas[2]
    # Fill class biases
    self.b_class = np.zeros(self.C)

  def to_keras(self,
               dropouts = [0.1,0.1],
               lr=0.05,
               decay=1e-5,
               momentum=0.9,
               nesterov=True,
               loss='categorical_crossentropy',
               ):
    self.input_layer = Input(shape=(self.D,))
    self.nodes_layer = Dense(self.N,
                        activation="tanh")(self.input_layer)

    self.drop_nodes_layer = Dropout(dropouts[0])(self.nodes_layer)
    self.leaves_layer = Dense(self.L,
                        activation="tanh")(self.drop_nodes_layer)
    self.drop_nodes_layer = Dropout(dropouts[1])(self.leaves_layer)
    self.output_layer = Dense(self.C, activation='softmax')(self.drop_nodes_layer)
    self.model = Model(input=self.input_layer, output=self.output_layer)
    self.model_nodes = Model(input=self.input_layer, output=self.nodes_layer)
    self.model_leaves = Model(input=self.input_layer, output=self.leaves_layer)

    self.sgd = optimizers.SGD(lr=lr,
               decay=decay,
               momentum=momentum,
               nesterov=nesterov)
    self.model.compile(loss=loss, optimizer=self.sgd)

    self.model.layers[1].set_weights(weights=[self.W_in_nodes,
                                             self.b_nodes.values.flatten()])
    self.model.layers[3].set_weights(weights=[self.W_nodes_leaves,
                                             self.b_leaves.values.flatten()])
    self.model.layers[5].set_weights(weights=[self.W_leaves_out,
                                              self.b_class.flatten()])

  def fit(self,
          X,
          y,
          epochs=100,
          min_delta=0,
          patience=0,
          ):
    y = to_categorical(y)
    early_stopping = EarlyStopping(monitor='loss',
                     min_delta=min_delta,
                     patience=patience,
                     verbose=0,
                     mode='auto')
    callbacks_list = [early_stopping]
    history = self.model.fit(x = X,
                             y = y,
                             callbacks = callbacks_list,
                             epochs = epochs,
                             verbose = 0)
    return history.history["loss"]

  def get_activations(self,
              X,
              y = None):
    return self.model_nodes.predict(X),self.model_leaves.predict(X), self.model.predict(X)

  def predict_class(self,
                    X,
                    y = None):
    class_indexes = np.argmax(self.model.predict(X),axis=1)
    return self.classes[class_indexes]

  def score(self,
            X,
            y):
    return accuracy_score(self.predict_class(X),y)

if __name__ == "__main__":
  import graphviz 
  import matplotlib.pyplot as plt
  # Build a dataset
  dataset_length = 100
  D = 2
  X = np.random.randn(dataset_length,D)*0.1
  X[0:dataset_length//2,0] += 0.1
  X[0:dataset_length//2,0] += 0.2
  Y = np.ones(dataset_length)
  Y[0:dataset_length//2] *= 0
  # Train a Tree
  clf = DecisionTreeClassifier()
  clf = clf.fit(X, Y)

  # plot the tree
  dot_data = export_graphviz(clf, out_file=None, 
                           feature_names=["feature_"+str(i) for i in range(D)],  
                           class_names=["class_0", "class_1"],  
                           filled=True, rounded=True,  
                           special_characters=True)  

  graph = graphviz.Source(dot_data)  
  graph.render('prarent_tree.gv', view=True) 
  print(get_list_split_phi(clf))

  rf = RandomForestClassifier(max_depth=5,n_estimators=100)
  rf.fit(X, Y)
  forest_splits = get_list_split_phi_forest(rf)

  splits_df = pd.DataFrame(forest_splits)

  X_test = np.random.randn(dataset_length,D)*0.1
  X_test[0:dataset_length//2,0] += 0.1
  X_test[0:dataset_length//2,0] += 0.2
  Y_test = np.ones(dataset_length)
  Y_test[0:dataset_length//2] *= 0

  a = dt2nn(clf,2,gammas=[100,100,100])
  a.to_keras()
  print a.score(X_test,Y_test)
  print a.score(X,Y)
  errors = a.fit(X,Y)

  print a.score(X_test, Y_test)
  print a.score(X,Y)

