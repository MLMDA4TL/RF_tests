# -*- coding: utf-8 -*-
"""
@author: sergio
"""

import numpy as np
from sklearn import tree
import copy
from collections import OrderedDict 

"""
children_left

children_right

feature

threshold 

value : counts of each class at each node

impurity

n_node_samples
"""



def get_list_split_phi(decision_tree,node=0):
  def rec_get_list_split_phi(decisiontree,splits_phi,node):
    phi = decisiontree.feature[node]
    threshold = decisiontree.threshold[node]
    if phi != -2:
      splits_phi[node] = [phi,threshold]
    children = [decisiontree.children_left[node],
                decisiontree.children_right[node]]
    for child in children:
      if child != -1:
        rec_get_list_split_phi(decisiontree,splits_phi,child)
    return 0
  splits_phi = {}
  rec_get_list_split_phi(decision_tree.tree_,splits_phi,node=node)
  return OrderedDict(splits_phi)

def get_parent_nodes_leaf(leaf,decision_tree):
  def bottom_up_get_parent(node, parents, tree):
    p,b = find_parent(tree, node)
    if b != 0:
      parents.append((p,b))
      bottom_up_get_parent(p, parents, tree)
  parents = []
  bottom_up_get_parent(leaf, parents, decision_tree.tree_)
  return parents

def get_parents_nodes_leaves_dic(decision_tree):
  leaves = leaves_id(decision_tree.tree_)
  leaves_parents = {}
  for leave in leaves:
    parents = get_parent_nodes_leaf(leave, decision_tree)
    leaves_parents[leave] = parents[:]
  return OrderedDict(leaves_parents)


def get_list_split_phi_forest(random_forest):
  list_split_phi = []
  for i,dtree in enumerate(random_forest.estimators_):
    list_split_phi_tree = get_list_split_phi(dtree)
    list_split_phi += list_split_phi_tree 
  return list_split_phi


def get_children_distributions(decisiontree,
                               node_index):
  tree = decisiontree.tree_
  child_l = tree.children_left[node_index]
  child_r = tree.children_right[node_index]
  Q_source_l = tree.value[child_l]
  Q_source_r = tree.value[child_r]
  return [np.asarray(Q_source_l), np.asarray(Q_source_r)]
                                   
def compute_Q_children(X_node,
                       Y_node,
                       phi,
                       threshold,
                       classes):
  # Split parent node target sample using the threshold provided
  # instances <= threshold go to the left 
  # instances > threshold go to the right 
  decision_l = X_node[:,phi] <= threshold
  decision_r = np.logical_not(decision_l)
  Y_target_child_l = Y_target_node[decision_l]
  Y_target_child_r = Y_target_node[decision_r]
  Q_l = compute_class_distribution(classes, Y_child_l)
  Q_r = compute_class_distribution(classes, Y_child_r) 
  return Q_l,Q_r

def get_node_distribution(decisiontree,
                          node_index):
  tree = decisiontree.tree_
  Q = tree.value[node_index]
  return np.asarray(Q)

  
def compute_class_distribution(classes,
                               class_membership):
  unique, counts = np.unique(class_membership,
                             return_counts=True)
  classes_counts = dict(zip(unique, counts))
  classes_index = dict(zip(classes,range(len(classes))))
  distribution = np.zeros(len(classes))
  for label,count in classes_counts.items():
      class_index = classes_index[label]
      distribution[class_index] = count
  return distribution

def KL_divergence(class_counts_P,
                  class_counts_Q):
  # KL Divergence to assess the difference between two distributions
  # Definition: $D_{KL}(P||Q) = \sum{i} P(i)ln(\frac{P(i)}{Q(i)})$
  # epsilon to avoid division by 0
  epsilon = 1e-8
  class_counts_P += epsilon
  class_counts_Q += epsilon
  P = class_counts_P * 1./ class_counts_P.sum()
  Q = class_counts_Q * 1./ class_counts_Q.sum()
  Dkl = (P * np.log(P * 1./ Q)).sum()
  return Dkl

def H(class_counts):
  # Entropy
  # Definition: $H(P) = \sum{i} -P(i) ln(P(i))$
  epsilon = 1e-8
  class_counts += epsilon
  P = class_counts * 1./ class_counts.sum()
  return - (P * np.log(P)).sum()

def IG(class_counts_parent,
       class_counts_children):
  # Information Gain
  H_parent = H(class_counts_parent)
  H_children = np.asarray([H(class_counts_child) for class_counts_child in class_counts_children])
  N = class_counts_parent.sum()
  p_children = np.asarray([class_counts_child.sum()*1./N for class_counts_child in class_counts_children])
  information_gain = H_parent - (p_children * H_children).sum()
  return information_gain
        
def JSD(P,Q):
  M = (P+Q) * 1./2
  Dkl_PM = KL_divergence(P,M)
  Dkl_QM = KL_divergence(Q,M)
  return (Dkl_PM + Dkl_QM) * 1./2

def DG(Q_source_l,
       Q_source_r,
       Q_target_l,
       Q_target_r):
  # compute proportion of instances at left and right
  p_l = Q_target_l.sum() 
  p_r = Q_target_r.sum() 
  total_counts = p_l + p_r
  p_l /= total_counts
  p_r /= total_counts
  # compute the DG
  return 1. - p_l * JSD(Q_target_l, Q_source_l) - p_r * JSD(Q_target_r, Q_source_r)



def prune_subtree(decisiontree,
                  node_index):
  tree = decisiontree.tree_
  if tree.children_left[node_index] != -1:
    prune_subtree(decisiontree,
                  tree.children_left[node_index])
    tree.children_left[node_index] = -1
  if tree.children_right[node_index] != -1:
    prune_subtree(decisiontree,
                  tree.children_right[node_index])
    tree.children_right[node_index] = -1

def GINI(class_distribution):
  if class_distribution.sum():
    p = class_distribution / class_distribution.sum()
    return 1 - (p**2).sum()
  return 0

def leaves_id(tree):
  return np.asarray(range(tree.children_right.size))[tree.children_left == tree.children_right]

def find_parent(tree, i_node):
  p = -1
  b = 0
  dic = tree.__getstate__()
  if i_node != 0 and i_node != -1:
    if i_node in dic['nodes']['left_child']:
      p = list(dic['nodes']['left_child']).index(i_node)
      b = -1
    elif i_node in dic['nodes']['right_child']:
      p = list(dic['nodes']['right_child']).index(i_node)
      b = 1
  return p, b