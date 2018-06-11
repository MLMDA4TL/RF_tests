# -*- coding: utf-8 -*-
"""
@author: sergio
"""
import matplotlib.pyplot as plt
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
from keras.regularizers import l1
BIAS = 1
DIM = 0


class ndt:
  def __init__(self,D,gammas=[10,1,1],tree_id=None,sigma=0):
    self.gammas = gammas
    self.D = D
    self.tree_id = tree_id
    self.sigma = sigma

  def compute_matrices_and_biases(self, decision_tree):
    self.splits = pd.DataFrame(get_list_split_phi(decision_tree)).T
    self.leaves = get_parents_nodes_leaves_dic(decision_tree)
    self.N = self.splits.shape[0]
    self.L = len(self.leaves)
    # Fill Input -> Nodes layer matrix
    self.W_in_nodes = pd.DataFrame(np.zeros((self.D,self.N)),
                                   index = list(range(self.D)),
                                   columns = self.splits.index)
    for node, dim in self.splits[DIM].iteritems():
      self.W_in_nodes.loc[dim,node] = 1.*self.gammas[0]
    # Fill Input -> Nodes layer biases
    self.b_nodes = pd.DataFrame(- self.splits[BIAS]*self.gammas[0])
    self.b_nodes.columns = ["NODES_BIASES"]
    # Fill Nodes -> Leaves layer matrix
    self.W_nodes_leaves = pd.DataFrame(np.zeros((self.N,self.L)),
                                      index = self.splits.index,
                                      columns = self.leaves.keys())
    for leave,node_sides in self.leaves.iteritems():
      for node,r_l in node_sides:
        self.W_nodes_leaves.loc[node,leave] = r_l*self.gammas[1]
    # Fill Nodes -> Leaves layer biases       
    b_leaves = {k:-len(x)+0.5 for k,x in self.leaves.iteritems()}
    self.b_leaves = pd.DataFrame(b_leaves.values(),
                                 index=b_leaves.keys(),
                                 columns = ["LEAVES_BIASES"])
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
    self.b_class = pd.DataFrame(np.zeros(self.C),
                                index = self.classes,
                                columns = ["CLASS_BIASES"])
    self.specify_tree_id()

  def specify_tree_id(self):
    def add_id_2_elements(list_values, id_to_add):
      append_id = lambda x:str(id_to_add)+"_"+str(x)
      return map(append_id,list_values)
    if self.tree_id != None:
      # Rename input -> nodes matrix biases
      self.W_in_nodes.columns = add_id_2_elements(self.W_in_nodes.columns,
                                                  self.tree_id)
      self.b_nodes.index = add_id_2_elements(self.b_nodes.index,self.tree_id)
      # Rename nodes -> leaves matrix biases
      self.W_nodes_leaves.index = add_id_2_elements(self.W_nodes_leaves.index,
                                                    self.tree_id)
      self.W_nodes_leaves.columns = add_id_2_elements(self.W_nodes_leaves.columns,
                                                      self.tree_id)
      self.b_leaves.index = add_id_2_elements(self.b_leaves.index,self.tree_id)
      # Rename leaves -> classes matrix biases
      self.W_leaves_out.index = add_id_2_elements(self.W_leaves_out.index,
                                                  self.tree_id)

  def to_keras(self,
               dropouts = [0.1,0.1,0.1],
               loss='categorical_crossentropy',
               optimizer = optimizers.Adam,
               kernel_regularizer=[l1(0),l1(0),l1(0)],
               optimizer_params = {"lr":0.001,
                                   "beta_1":0.9,
                                   "beta_2":0.999,
                                   "epsilon":1e-8,
                                   "decay":1e-6}):
    self.input_layer = Input(shape=(self.D,))
    self.drop_input_layer = Dropout(dropouts[0])(self.input_layer)

    self.nodes_layer = Dense(self.N,
                        activation="tanh",
                        kernel_regularizer=kernel_regularizer[0])(self.drop_input_layer)

    self.drop_nodes_layer = Dropout(dropouts[1])(self.nodes_layer)

    self.leaves_layer = Dense(self.L,
                        activation="tanh",
                        kernel_regularizer=kernel_regularizer[1])(self.drop_nodes_layer)
    
    self.drop_nodes_layer = Dropout(dropouts[2])(self.leaves_layer)

    self.output_layer = Dense(self.C,
                              activation='softmax',
                              kernel_regularizer=kernel_regularizer[2])(self.drop_nodes_layer)

    self.model = Model(input=self.input_layer, output=self.output_layer)
    self.model_nodes = Model(input=self.input_layer, output=self.nodes_layer)
    self.model_leaves = Model(input=self.input_layer, output=self.leaves_layer)

    self.sgd = optimizer(**optimizer_params)
    self.model.compile(loss=loss, optimizer=self.sgd)
    
    flat_b_nodes = self.b_nodes.values.flatten()
    flat_b_leaves = self.b_leaves.values.flatten()
    flat_b_class = self.b_class.values.flatten()

    self.model.layers[2].set_weights(weights=[self.W_in_nodes+np.random.randn(*self.W_in_nodes.shape)*self.sigma,
                                             flat_b_nodes+np.random.randn(*flat_b_nodes.shape)*self.sigma])
    self.model.layers[4].set_weights(weights=[self.W_nodes_leaves+np.random.randn(*self.W_nodes_leaves.shape)*self.sigma,
                                             flat_b_leaves+np.random.randn(*flat_b_leaves.shape)*self.sigma])
    self.model.layers[6].set_weights(weights=[self.W_leaves_out+np.random.randn(*self.W_leaves_out.shape)*self.sigma,
                                              flat_b_class+np.random.randn(*flat_b_class.shape)*self.sigma])

  def fit(self,
          X,
          y,
          epochs=100,
          min_delta=0,
          patience=10,
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
    nodes_a = pd.DataFrame(self.model_nodes.predict(X),
                           columns = self.b_nodes.index)
    leaves_a = pd.DataFrame(self.model_leaves.predict(X),
                            columns = self.b_leaves.index)
    output_a = pd.DataFrame(self.model.predict(X),
                            columns = self.b_class.index)
    return nodes_a, leaves_a, output_a

  def get_weights_from_NN(self):
    w_2 = self.model.layers[2].get_weights()
                       
    self.W_in_nodes_nn = pd.DataFrame(w_2[0],
                         index = self.W_in_nodes.index,
                         columns = self.W_in_nodes.columns)
    self.b_nodes_nn = pd.DataFrame(w_2[1],
                                index=self.b_nodes.index,
                                columns=self.b_nodes.columns)

    w_4 = self.model.layers[4].get_weights()
    self.W_nodes_leaves_nn = pd.DataFrame(w_4[0],
                             index = self.W_nodes_leaves.index,
                             columns = self.W_nodes_leaves.columns)
    self.b_leaves_nn = pd.DataFrame(w_4[1],
                                    index=self.b_leaves.index,
                                    columns=self.b_leaves.columns)

    w_6 = self.model.layers[6].get_weights()
    self.W_leaves_out_nn = pd.DataFrame(w_6[0],
                                        index = self.W_leaves_out.index,
                                        columns = self.W_leaves_out.columns)
    self.b_class_nn = pd.DataFrame(w_6[1],
                                   index=self.b_class.index,
                                   columns=self.b_class.columns)

  def compute_weights_differences(self):
    self.get_weights_from_NN()
    self.diff_W_in_nodes = self.W_in_nodes - self.W_in_nodes_nn
    self.diff_b_nodes = self.b_nodes - self.b_nodes_nn
    self.diff_W_nodes_leaves = self.W_nodes_leaves - self.W_nodes_leaves_nn
    self.diff_b_leaves = self.b_leaves - self.b_leaves_nn
    self.diff_W_leaves_output = self.W_leaves_out - self.W_leaves_out_nn
    self.diff_b_class = self.b_class - self.b_class_nn

  def predict_class(self,
                    X,
                    y = None):
    class_indexes = np.argmax(self.model.predict(X),axis=1)
    return self.classes[class_indexes]

  def score(self,
            X,
            y):
    return accuracy_score(self.predict_class(X),y)

  def plot_differences(self):
    if "diff_W_in_nodes" not in dir(self):
      self.compute_weights_differences()
    fig=plt.figure(figsize=(3, 2))
    columns = 3
    rows = 2
    ax1a = fig.add_subplot(rows, columns, 1)
    plt.imshow(self.diff_W_in_nodes,aspect="auto",cmap="gray")
    ax1a.set_title("diff W in nodes")

    ax2a = fig.add_subplot(rows, columns, 2)
    plt.imshow(self.diff_b_nodes,aspect="auto",cmap="gray")
    ax2a.set_title("diff b nodes")

    ax3a = fig.add_subplot(rows, columns, 3)
    plt.imshow(self.diff_W_nodes_leaves,aspect="auto",cmap="gray")
    ax3a.set_title("diff W nodes leaves")

    ax4a = fig.add_subplot(rows, columns, 4)
    plt.imshow(self.diff_b_leaves,aspect="auto",cmap="gray")
    ax4a.set_title("diff b leaves")

    ax5a = fig.add_subplot(rows, columns, 5)
    plt.imshow(self.diff_W_leaves_output,aspect="auto",cmap="gray")
    ax5a.set_title("diff W leaves out")

    ax6a = fig.add_subplot(rows, columns, 6)
    plt.imshow(self.diff_b_class,aspect="auto",cmap="gray")
    ax6a.set_title("diff b class")  
    plt.show()

  def plot_W_nn_quantiles(self,quantiles = np.arange(0,99.999,0.001)):
    fig=plt.figure(figsize=(3, 2))
    columns = 3
    rows = 2
    ax1a = fig.add_subplot(rows, columns, 1)
    plt.plot(quantiles,np.percentile(self.W_in_nodes_nn,quantiles))
    plt.plot(quantiles,np.percentile(self.W_in_nodes,quantiles))
    ax1a.set_title("W in nodes")

    ax2a = fig.add_subplot(rows, columns, 2)
    plt.plot(quantiles,np.percentile(self.b_nodes_nn,quantiles))
    plt.plot(quantiles,np.percentile(self.b_nodes,quantiles))
    ax2a.set_title("b nodes")

    ax3a = fig.add_subplot(rows, columns, 3)
    plt.plot(quantiles,np.percentile(self.W_nodes_leaves_nn,quantiles))
    plt.plot(quantiles,np.percentile(self.W_nodes_leaves,quantiles))
    ax3a.set_title("W nodes leaves")

    ax4a = fig.add_subplot(rows, columns, 4)
    plt.plot(quantiles,np.percentile(self.b_leaves_nn,quantiles))
    plt.plot(quantiles,np.percentile(self.b_leaves,quantiles))
    ax4a.set_title("b leaves")

    ax5a = fig.add_subplot(rows, columns, 5)
    plt.plot(quantiles,np.percentile(self.W_leaves_out_nn,quantiles))
    plt.plot(quantiles,np.percentile(self.W_leaves_out,quantiles))
    ax5a.set_title("W leaves out")

    ax6a = fig.add_subplot(rows, columns, 6)
    plt.plot(quantiles,np.percentile(self.b_class_nn,quantiles))
    plt.plot(quantiles,np.percentile(self.b_class,quantiles))
    ax6a.set_title("b class")  
    plt.show()    


  def print_tree_weights(self):
    print("W: Input -> Nodes")
    print(self.W_in_nodes)
    print("b: Input -> Nodes")
    print(self.b_nodes)
    print("W: Nodes -> Leaves")
    print(self.W_nodes_leaves)
    print("b: Nodes -> Leaves")
    print(self.b_leaves)
    print("W: Leaves -> Out")
    print(self.W_leaves_out)
    print("b: Leaves -> Out")
    print(self.b_class)

  def print_nn_weights(self):
    print("W: Input -> Nodes")
    print(self.W_in_nodes_nn)
    print("b: Input -> Nodes")
    print(self.b_nodes_nn)
    print("W: Nodes -> Leaves")
    print(self.W_nodes_leaves_nn)
    print("b: Nodes -> Leaves")
    print(self.b_leaves_nn)
    print("W: Leaves -> Out")
    print(self.W_leaves_out_nn)
    print("b: Leaves -> Out")
    print(self.b_class_nn)

  def plot_old_new_network(self):
    if "W_in_nodes" not in dir(self):
      self.get_weights_from_NN()
    fig=plt.figure(figsize=(6, 2))
    columns = 6
    rows = 2
    ax1a = fig.add_subplot(rows, columns, 1)
    plt.imshow(self.W_in_nodes,aspect="auto",cmap="gray")
    ax1a.set_title("W in nodes")
    ax1b = fig.add_subplot(rows, columns, 2)
    plt.imshow(self.W_in_nodes_nn,aspect="auto",cmap="gray")
    ax1b.set_title("W in nodes nn")

    ax2a = fig.add_subplot(rows, columns, 3)
    plt.imshow(self.b_nodes,aspect="auto",cmap="gray")
    ax2a.set_title("b nodes ")
    ax2b = fig.add_subplot(rows, columns, 4)
    plt.imshow(self.b_nodes_nn,aspect="auto",cmap="gray")
    ax2b.set_title("b nodes nn")

    ax3a = fig.add_subplot(rows, columns, 5)
    plt.imshow(self.W_nodes_leaves,aspect="auto",cmap="gray")
    ax3a.set_title("W nodes leaves")
    ax3b = fig.add_subplot(rows, columns, 6)
    plt.imshow(self.W_nodes_leaves_nn,aspect="auto",cmap="gray")
    ax3b.set_title("W nodes leaves nn")

    ax4a = fig.add_subplot(rows, columns, 7)
    plt.imshow(self.b_leaves,aspect="auto",cmap="gray")
    ax4a.set_title("b leaves")
    ax4b = fig.add_subplot(rows, columns, 8)
    plt.imshow(self.b_leaves_nn,aspect="auto",cmap="gray")
    ax4b.set_title("b leaves nn")

    ax5a = fig.add_subplot(rows, columns, 9)
    plt.imshow(self.W_leaves_out,aspect="auto",cmap="gray")
    ax5a.set_title("W leaves out")
    ax5b = fig.add_subplot(rows, columns, 10)
    plt.imshow(self.W_leaves_out_nn,aspect="auto",cmap="gray")
    ax5b.set_title("W leaves out nn")

    ax6a = fig.add_subplot(rows, columns, 11)
    plt.imshow(self.b_class,aspect="auto",cmap="gray")
    ax6a.set_title("b class")
    ax6b = fig.add_subplot(rows, columns, 12)
    plt.imshow(self.b_class_nn,aspect="auto",cmap="gray")
    ax6b.set_title("b class nn")    
    plt.show()

if __name__ == "__main__":
  # Build a dataset
  import matplotlib.pyplot as plt
  dataset_length = 10000
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
  clf = DecisionTreeClassifier(max_depth=20)
  clf = clf.fit(X, Y)

  a = ndt(D=2,gammas=[10,1,1],tree_id=0)
  a.compute_matrices_and_biases(clf)
  a.to_keras(dropouts=[0.1,0.5,0.5])
  print "scores before training"
  print a.score(X_test,Y_test)
  print a.score(X,Y)

  print clf.score(X_test,Y_test)
  print clf.score(X,Y)
  errors = a.fit(X,Y,epochs=1000)
  print "scores after training"
  print a.score(X_test, Y_test)
  print a.score(X,Y)
  a.get_weights_from_NN()
  print "Tree weights"
  a.print_tree_weights()
  print "NN weights"
  a.print_nn_weights()
  #print "activations"
  #print a.get_activations(X)
  differences = a.compute_weights_differences()


