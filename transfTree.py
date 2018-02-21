#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:43:15 2018

@author: mounir
"""

import os
import sys
#import glob
import numpy as np
import sklearn
import copy

import scipy 
import signal_lib as sl
import clf_lib as clib
import exp_lib as expl

import sklearn.ensemble as skl_ens
import sklearn.externals.joblib as jb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import sklearn.linear_model as skl_linear_model


from sklearn import datasets

#==============================================================================
# 
#==============================================================================


#important à maitriser :
def fusionTree(tree1, f, tree2):
    dic = tree1.__getstate__()
    dic2 = tree2.__getstate__()
    
    size_init = tree1.node_count
    
    tree1.max_depth = tree1.max_depth + tree2.max_depth 
    tree1.capacity = tree1.capacity + tree2.capacity - 1
    tree1.node_count = tree1.node_count + tree2.node_count - 1
    
    dic['nodes'][f] = dic2['nodes'][0]
    dic['nodes']['left_child'][f] = dic2['nodes']['left_child'][0]
    dic['nodes']['right_child'][f] = dic2['nodes']['right_child'][0]

    #impurity
    #Laisser chgt values , weighted_n_node_samples & n_node_samples
    
    dic['nodes'] = np.concatenate((dic['nodes'] , dic2['nodes'][1:]  ))
    dic['nodes']['left_child'][size_init:] = dic['nodes']['left_child'][size_init:] + size_init - 1
    dic['nodes']['right_child'][size_init:] = dic['nodes']['right_child'][size_init:] + size_init - 1

    return tree1
    
def fusionDecisionTree(dTree1, f, dTree2):
    dTree1.max_depth = dTree1.tree_.max_depth + dTree2.tree_.max_depth 
    fusionTree(dTree1.tree_,f,dTree2.tree_)
    return dTree1

def updateValues(tree,values):
    tree.__getstate__()['values'] = values
  
    
def leaf_error(tree,node):
    if node == -1 :
        return 0
    else:
        return np.min(tree.value[node])/np.sum(tree.value[node])

def error(tree,node):
    if node == -1 : 
        return 0,0
    else: 
    
        if tree.feature[node] == -2 :
            return leaf_error(tree,node),1
        else:
            er,nr = error(tree,tree.children_right[node])
            el,nl = error(tree,tree.children_left[node])
            return (el*nl+er*nr)/(nl+nr),nl+nr

def cut_into_leaves_tree(tree,nodes):
    dic = tree.__getstate__()
    node_to_rem = list()
    for node in nodes:
        size_init = tree.node_count
        i = node
        j = node
        
        while(tree.feature[i] != -2 or tree.feature[j] != -2):
            if tree.feature[i] != -2 :
                i = tree.children_left[node]
            node_to_rem.append(i)
            if tree.feature[j] != -2:
                j = tree.children_right[node]
            node_to_rem.append(j)
        
        tree.capacity = tree.capacity - len(node_to_rem)
        tree.node_count = tree.node_count - len(node_to_rem)
                
    dic['nodes'] = dic['nodes'][list(set(np.linspace(0,size_init-1,size_init).astype(int))-set(node_to_rem))]
    
        #Attention Max_depth
        #tree.max_depth =

        
    
    return tree

def cut_into_leaves(dTree,nodes):
    cut_into_leaves_tree(dTree.tree_,nodes)
    #attention max_depth


def SER(dTree,X_target,y_target):
    
    t_copy = copy.deepcopy(dTree.tree_)
    target_values = np.zeros((t_copy.node_count,1,t_copy.value.shape[-1]))
    
    sparseM = dTree.decision_path(X_target)
    #target_values[:,:,0] = np.sum(sparseM.toarray()[np.where(y_target == 0)[0]],axis = 0)
    #target_values[:,:,1] = np.sum(sparseM.toarray()[np.where(y_target == 1)[0]],axis = 0)

    leaves = np.where(t_copy.feature == -2)[0]

    #expansion
    print('Expansion...')
    for f in leaves :
        ind = np.where(sparseM.toarray()[:,f] == 1 )[0]
        
        if ind.size != 0 :
            Sv = X_target[ind]
            
            #build full new tree from f
            DT_to_add = sklearn.tree.DecisionTreeClassifier()
            DT_to_add.min_impurity_split = 0
            DT_to_add.fit(Sv,y_target[ind])
            fusionDecisionTree(dTree, f, DT_to_add)
        
    sparseM = dTree.decision_path(X_target)
    for iy in target_values.shape[-1]:
        target_values[:,0,iy] = np.sum(sparseM.toarray()[np.where(y_target == iy)[0]],axis = 0)
   
    updateValues(dTree.tree_,target_values)
    print(str(dTree.tree_.node_count)+' noeuds apres expansion')
    #t_copy = copy.deepcopy(dTree.tree_)
    #print(error(dTree.tree_,0))
    #reduction
    print('Reduction...')
    node_to_cut = list()
    for i_node in range(dTree.tree_.node_count):

        le = leaf_error(dTree.tree_,i_node )
        e,nn= error(dTree.tree_,i_node)
        if le <= e :
            node_to_cut.append(i_node)
    
    cut_into_leaves(dTree,node_to_cut)

    return dTree

iris = datasets.load_iris()
X = iris.data  
y = iris.target

K = 3
c = 0

rf_s = list()
ind_test = list()
kf = StratifiedKFold(n_splits=K) 
for train, test in kf.split(X,y):  
    c = c+1      
    #print(c)
    rf_s.append(skl_ens.RandomForestClassifier(n_estimators=50,max_depth=10, oob_score=True))
    rf_s[-1].fit(X[train],y[train])
    ind_test.append(test)

d = rf_s[0].estimators_[0]
print('Arbre initial appr. sur '+str(train.size)+' données : '+str(d.tree_.node_count)+' noeuds')
#SER(d,X[ind_test[0]],y[ind_test[0]])
#print('Arbre initial réajuté sur '+str(test.size)+' données : '+str(d.tree_.node_count)+' noeuds')
