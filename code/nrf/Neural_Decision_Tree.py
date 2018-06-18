# -*- coding: utf-8 -*-
"""
@author: sergio
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from collections import OrderedDict 
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras import optimizers
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from keras.regularizers import l1
sys.path.append("../utils")
from common_functions import find_parent,leaves_id,get_list_split_phi_forest,get_list_split_phi,get_parents_nodes_leaves_dic, print_decision_path

BIAS = 1
DIM = 0
RIGHT = 1
LEFT = -1
LEAVES = -1
LEAVES_THRESHOLDS = -2
LEAVES_FEATURES = -2
EMPTY_NODE = -5

class ndt:
  def __init__(self,D,gammas=[10,1,1],tree_id=None,sigma=0):
    """
    Creates and neural decision tree
    :param gammas: Metaparameter for each layer of the neural decision tree (slope of the tanh function). High gamma -> behavior of NN is closer to tree (and also harder to change).
    :param tree_id: identifier for the tree.
    :param sigma: STD for the initial weights
    """
    self.gammas = gammas
    self.D = D
    self.tree_id = tree_id
    self.sigma = sigma

  def compute_matrices_and_biases(self, decision_tree):
    """
    Compute synaptic weights and biases according to a decision tree 
    :param decision_tree: scikit-learn decision tree
    """
    self.decision_tree = decision_tree
    self.splits = pd.DataFrame(get_list_split_phi(decision_tree)).T
    self.leaves = get_parents_nodes_leaves_dic(decision_tree)
    self.N = self.splits.shape[0]
    self.L = len(self.leaves)
    # Fill Input -> Nodes layer matrix
    self.W_in_nodes = pd.DataFrame(np.zeros((self.D,self.N)),
                                   index = list(range(self.D)),
                                   columns = self.splits.index)
    for node, dim in self.splits[DIM].iteritems():
      self.W_in_nodes.loc[dim,node] = 1.
    self.W_in_nodes *= self.gammas[0]
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
    """
    Change the name of the columns/indexes of the weights and biases to  include the tree id
    """
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
    """
    Creates a keras neural network
    :param dropouts: Dropouts for each layer
    :param loss: loss function
    :param optimizer: keras optimizer
    :param kernel_regularizer: regularization constrains for each layer
    :param optimizer_params: dictionnary of parameters for the optimizer
    """
    self.input_layer = Input(shape=(self.D,))
    self.drop_input_layer = Dropout(dropouts[0])(self.input_layer)

    self.nodes_layer = Dense(self.N,
                       activation="tanh",
                       kernel_regularizer=kernel_regularizer[0])(self.drop_input_layer)

    self.drop_nodes_layer = Dropout(dropouts[1])(self.nodes_layer)

    self.leaves_layer = Dense(self.L,
                        activation="softmax",
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
          to_categorical_conversion=True,
          ):
    """
    Fit the neural decision tree
    :param X: Training set
    :param y: training set labels
    :param epochs: number of epochs
    :param min_delta: stoping criteria delta 
    :param patience: stoping criteria patience
    """
    if to_categorical_conversion:
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
    """
    Get the activation for each layer
    :param X: data set
    :param y: labels
    """
    nodes_a = pd.DataFrame(self.model_nodes.predict(X),
                           columns = self.b_nodes.index)
    leaves_a = pd.DataFrame(self.model_leaves.predict(X),
                            columns = self.b_leaves.index)
    output_a = pd.DataFrame(self.model.predict(X),
                            columns = self.b_class.index)
    return nodes_a, leaves_a, output_a

  def get_weights_from_NN(self):
    """
    Get the weights from the keras NN, and load them into attributes of the neural decision tree
    """
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
    """
    Computes the difference between the original tree weights and those after training
    """
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
    """
    Predict class membership with the neural decision tree
    :param X: dataset 
    :param y: labels
    """
    class_indexes = np.argmax(self.model.predict(X),axis=1)
    return self.classes[class_indexes]

  def score(self,
            X,
            y):
    """
    Compute prediction score
    :param X: dataset
    :param y: labels
    """
    return accuracy_score(self.predict_class(X),y)

  def plot_differences(self):
    """
    Plot the difference between the original tree weights and those after training
    """
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
    """
    Plot the weights and biases quantiles
    """
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

  def neural_network_to_tree(self,node_leaves_matrix=None,in_nodes_matrix=None,nodes_biases=None,threshold=0.9):
    def insert_node(parent,
                    right_left,
                    children_right,
                    children_left,
                    node_leaves_matrix,
                    threshold):
      
      print "_____________"
      print "parent",parent
      print "right_left",right_left
      print "nodes_leaves_sum"
      print node_leaves_matrix
      

      # Find parent's child
      node_leaves_matrix = node_leaves_matrix[np.abs(node_leaves_matrix).sum(axis=1)>0]
      count_nb_leaves = np.abs(node_leaves_matrix).sum(axis=1)
      
      child = count_nb_leaves.argmax()
      print "child",child,count_nb_leaves[child]

      # Remaining nodes are those that have at least one leaf
      remaining_nodes = list(count_nb_leaves.index[count_nb_leaves>0])
      
      print "nodes_leaves_sum"
      print np.abs(node_leaves_matrix).sum(axis=1)
      print "remaining nodes"
      print remaining_nodes
      
      # Add child to tree
      if parent is not None:
        if right_left == RIGHT:
          children_right[parent] = child
        elif right_left == LEFT:
          children_left[parent] = child

      # get leaves of current child
      current_node_leaves = node_leaves_matrix.loc[child]
      print "current node leaves"
      print current_node_leaves
      # Remove nodes that have been already included
      if parent in remaining_nodes:
        remaining_nodes.remove(parent)
      if child in remaining_nodes:
        remaining_nodes.remove(child)

      if remaining_nodes:
        node_leaves_matrix = node_leaves_matrix.loc[remaining_nodes]
      leaves = node_leaves_matrix.columns
      right_leaves = leaves[(current_node_leaves >= threshold).values]
      left_leaves = leaves[(current_node_leaves <= -threshold).values]
      
      print "leaves", leaves
      print "right_leaves", right_leaves 
      print "left_leaves", left_leaves
      
      # Apply the same method to right and left children 
      right_node_leaves_matrix = None
      if len(remaining_nodes):
        right_node_leaves_matrix = node_leaves_matrix[right_leaves]
        if np.abs(right_node_leaves_matrix).sum().sum() <= threshold:
          right_node_leaves_matrix = None
        else:
          insert_node(child,
                      RIGHT,
                      children_right,
                      children_left,
                      right_node_leaves_matrix,
                      threshold)
      if right_node_leaves_matrix is None:
        if not len(right_leaves):
          children_right[child] = EMPTY_NODE
        else:
          children_right[child] = right_leaves[0]
          if len(right_leaves) > 1:
            print "more than one leaf",right_leaves
          print "adding to", child, "children right",right_leaves 

      left_node_leaves_matrix = None
      if len(remaining_nodes):
        left_node_leaves_matrix = node_leaves_matrix[left_leaves]
        if np.abs(left_node_leaves_matrix).sum().sum() <= threshold:
          left_node_leaves_matrix = None
        else:
          insert_node(child,
                      LEFT,
                      children_right,
                      children_left,
                      left_node_leaves_matrix,
                      threshold)

      if left_node_leaves_matrix is None:
        if not len(left_leaves):
          children_left[child] = EMPTY_NODE
        else:
          children_left[child] = left_leaves[0]
          if len(left_leaves) > 1:
            print "more than one leaf",left_leaves
        print "adding to", child, "children left",left_leaves 

    if node_leaves_matrix is None:
      node_leaves_matrix = self.W_nodes_leaves
    if in_nodes_matrix is None:
      in_nodes_matrix = self.W_in_nodes
    if nodes_biases is None:
      nodes_biases = self.b_nodes

    children_right = pd.Series(LEAVES,index=list(node_leaves_matrix.index)+list(node_leaves_matrix.columns))
    children_left = pd.Series(LEAVES,index=list(node_leaves_matrix.index)+list(node_leaves_matrix.columns))
    thresholds = pd.Series(LEAVES_THRESHOLDS,index=list(node_leaves_matrix.index)+list(node_leaves_matrix.columns))
    features = pd.Series(LEAVES_FEATURES,index=list(node_leaves_matrix.index)+list(node_leaves_matrix.columns))
    node_leaves_matrix_local = np.copy(node_leaves_matrix)
    node_leaves_matrix_local = node_leaves_matrix_local * (np.abs(node_leaves_matrix)>=threshold*self.gammas[1])
    insert_node(None,
                None,
                children_right,
                children_left,
                node_leaves_matrix_local,
                threshold)
    # Compute the threshold of each node
    #print features
    #print thresholds
    #print nodes_biases
    thresholds[nodes_biases.index] = nodes_biases["NODES_BIASES"]*1./self.gammas[0]
    # Retrieve the most important feature for each node (one could imagine to create new features if necessary)
    features[in_nodes_matrix.columns] = in_nodes_matrix.index[in_nodes_matrix.values.argmax(axis=0)]
    return children_right, children_left, thresholds, features

  def assert_sample(self,X):
    """
    Print the decision path for the samples as well as the activation functions.
    :param X: dataset
    """
    print_decision_path(self.decision_tree,X)
    print(self.get_activations(X))

  def print_tree_weights(self):
    """
    Print tree weights
    """
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
    """
    Print NN weights
    """
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
    """
    Plot new and old network side by side
    """
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
  from sklearn.tree import DecisionTreeClassifier,export_graphviz
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
  clf = DecisionTreeClassifier(max_depth=10)
  clf = clf.fit(X, Y)

  a = ndt(D=2,gammas=[1,5,1],tree_id=0)
  a.compute_matrices_and_biases(clf)
  a.to_keras(dropouts=[0.,0.,0.])
  a.fit(X,Y)
  a.get_weights_from_NN()
  children_right, children_left, thresholds, features = a.neural_network_to_tree()

  children_right, children_left, thresholds, features = a.neural_network_to_tree(a.W_nodes_leaves_nn,a.W_in_nodes_nn,a.b_nodes_nn,0.9)
  #
  """
  children_right, children_left, thresholds, features = a.neural_network_to_tree()
  children_right.index = [int(val.split("_")[-1]) for val in children_right.index]
  children_right = children_right.sort_index()
  children_left.index = [int(val.split("_")[-1]) for val in children_left.index]
  children_left = children_left.sort_index()
  thresholds.index = [int(val.split("_")[-1]) for val in thresholds.index]
  thresholds = thresholds.sort_index()
  features.index = [int(val.split("_")[-1]) for val in features.index]
  features = features.sort_index()
  
  np.savetxt("left_true.csv",clf.tree_.children_left.astype(int), fmt='%i')
  np.savetxt("right_true.csv",clf.tree_.children_right.astype(int), fmt='%i')
  np.savetxt("feature_true.csv",clf.tree_.feature.astype(int), fmt='%i')
  np.savetxt("thresholds_true.csv",clf.tree_.threshold, fmt='%f')
  children_left.to_csv("try_left.csv")
  children_right.to_csv("try_right.csv")
  thresholds.to_csv("try_thresholds.csv")
  features.to_csv("try_features.csv")
  """

  print "scores before training"
  print a.score(X_test,Y_test)
  print a.score(X,Y)

  print clf.score(X_test,Y_test)
  print clf.score(X,Y)
  errors = a.fit(X,Y,epochs=10)
  print "scores after training"
  print a.score(X_test, Y_test)
  print a.score(X,Y)
  
  print "Tree weights"
  a.print_tree_weights()
  print "NN weights"
  a.print_nn_weights()
  #print "activations"
  #print a.get_activations(X)
  differences = a.compute_weights_differences()
  a.plot_differences()
  a.plot_old_new_network()
  
  import networkx as nx
  from networkx.drawing.nx_agraph import write_dot
  def plot_tree_like(children_left, children_right):
    G = nx.DiGraph()
    for node in children_right.index:
      G.add_node(node)
    for i,parent in enumerate(children_right.index):
      child = children_right[parent]
      if child == LEAVES or child == EMPTY_NODE:
        child = str(child)+"_"+str(i)
      G.add_edge(parent,child)
    for i,parent in enumerate(children_left.index):
      child = children_left[parent]
      if child == LEAVES or child == EMPTY_NODE:
        child = str(child)+"_"+str(i)
      G.add_edge(parent,child)
    return G
  children_right, children_left, thresholds, features = a.neural_network_to_tree(a.W_nodes_leaves_nn,a.W_in_nodes_nn,a.b_nodes_nn,0.9)
  G = plot_tree_like(children_left, children_right)
  write_dot(G,"test.dot")

  children_right, children_left, thresholds, features = a.neural_network_to_tree(threshold=0.1)
  G = plot_tree_like(children_left, children_right)
  write_dot(G,"test_prev.dot")





