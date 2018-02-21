#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:43:15 2018

@author: mounir
"""

import numpy as np
import sklearn
import copy


import sklearn.ensemble as skl_ens
import sklearn.externals.joblib as jb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import sklearn.linear_model as skl_linear_model

import pickle

from sklearn import datasets

#==============================================================================
# 
#==============================================================================

def updateValues(tree,values):
    d = tree.__getstate__()
    d['values'] = values
    tree.__setstate__(d)
   
def depth(tree,f):
    if f == -1:
        return 0
    if f == 0:
        return 0
    else:
        return max(depth(tree,tree.children_left[f]),depth(tree,tree.children_right[f]))+1
    
def depth_array(tree,inds):
    depths = np.zeros(np.array(inds).size)
    for i,e in enumerate(inds):
        depths[i] = depth(tree,i)
    return depths
    
        
#important à maitriser :
def fusionTree(tree1, f, tree2):
    dic = tree1.__getstate__().copy()
    dic2 = tree2.__getstate__()
    
    size_init = tree1.node_count
    
    if depth(tree1,f) +  dic2['max_depth'] > dic['max_depth']:
        dic['max_depth'] = depth(tree1,f) + tree2.max_depth 
    
    dic['capacity'] = tree1.capacity + tree2.capacity - 1
    dic['node_count'] = tree1.node_count + tree2.node_count - 1
    
    dic['nodes'][f] = dic2['nodes'][0]
    
    if (dic2['nodes']['left_child'][0] != - 1):
        dic['nodes']['left_child'][f] = dic2['nodes']['left_child'][0] + size_init - 1
    else:
        dic['nodes']['left_child'][f] = -1
    if (dic2['nodes']['right_child'][0] != - 1):
        dic['nodes']['right_child'][f] = dic2['nodes']['right_child'][0] + size_init - 1
    else:
        dic['nodes']['right_child'][f] = -1

    #impurity
    #Laisser chgt values , weighted_n_node_samples & n_node_samples
    
    dic['nodes'] = np.concatenate((dic['nodes'] , dic2['nodes'][1:]  ))
    dic['nodes']['left_child'][size_init:] = (dic['nodes']['left_child'][size_init:] != -1 )*(dic['nodes']['left_child'][size_init:] + size_init ) -1
    dic['nodes']['right_child'][size_init:] = (dic['nodes']['right_child'][size_init:] != -1 )*(dic['nodes']['right_child'][size_init:] + size_init) - 1

    # on agrandit également ce vecteur ( mais on le modifiera à l'extérieur)
    values = np.zeros((dic['node_count'],dic['values'].shape[1],dic['values'].shape[2]))
    dic['values'] = values
    
    #Attention :: (potentiellement important) 
    (Tree,(n_f,n_c,n_o),b) = tree1.__reduce__()
    del tree1
    tree1 = Tree(n_f,n_c,n_o) 
    
    tree1.__setstate__(dic)
    return tree1
    
def fusionDecisionTree(dTree1, f, dTree2):
    dTree1.tree_ = fusionTree(dTree1.tree_,f,dTree2.tree_)
    dTree1.max_depth = dTree1.tree_.max_depth
    return dTree1

  
    
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
        
def sub_nodes(tree,node):
    if ( node == -1 ):
        return list()
    if ( tree.feature[node] == -2 ):
        return [node]
    else:
        return [node] + sub_nodes(tree,tree.children_left[node]) + sub_nodes(tree,tree.children_right[node])
    
def cut_into_leaves_tree(tree,nodes):
    dic = tree.__getstate__().copy()
    node_to_rem = list()
    nodes = list(set(nodes))
    for node in nodes:
        size_init = tree.node_count
#        i = node
#        j = node
        
        node_to_rem = node_to_rem + sub_nodes(tree,node)[1:]
#        while(tree.feature[i] != -2 or tree.feature[j] != -2):
#            print('i : ',i)
#            print('j :',j)
#            if tree.feature[i] != -2 :
#                i = tree.children_left[node]
#            node_to_rem.append(i)
#            if tree.feature[j] != -2:
#                j = tree.children_right[node]
#            node_to_rem.append(j)
        
    node_to_rem = list(set(node_to_rem))
    depths = depth_array(tree,list(set(np.linspace(0,size_init-1,size_init).astype(int))-set(node_to_rem)))
    dic['max_depth'] = np.max(depths)
    
    dic['capacity'] = tree.capacity - len(node_to_rem)
    dic['node_count'] = tree.node_count  - len(node_to_rem)
    
    dic['nodes']['feature'][nodes] = -2
    dic['nodes']['left_child'][nodes] = -1
    dic['nodes']['right_child'][nodes] = -1
                
    dic['nodes'] = dic['nodes'][list(set(np.linspace(0,size_init-1,size_init).astype(int))-set(node_to_rem))]
    dic['values']= dic['values'][list(set(np.linspace(0,size_init-1,size_init).astype(int))-set(node_to_rem))]

    (Tree,(n_f,n_c,n_o),b) = tree.__reduce__()
    del tree
    
    tree = Tree(n_f,n_c,n_o) 
    tree.__setstate__(dic)
    
    return tree

def cut_into_leaves(dTree,nodes):
    dTree.tree_ = cut_into_leaves_tree(dTree.tree_,nodes)
    dTree.max_depth = dTree.tree_.max_depth


def SER(dTree,X_target,y_target):
    
    t_copy = copy.deepcopy(dTree.tree_)    
    sparseM = dTree.decision_path(X_target)
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
        
    t_copy = copy.deepcopy(dTree.tree_)
    target_values = np.zeros((t_copy.node_count,1,t_copy.value.shape[-1]))
    sparseM = dTree.decision_path(X_target)
    
    for iy in target_values.shape[-1]:
        target_values[:,0,iy] = np.sum(sparseM.toarray()[np.where(y_target == iy)[0]],axis = 0)
   
    updateValues(dTree.tree_,target_values)
    print(str(dTree.tree_.node_count)+' noeuds apres expansion')

    #reduction
    print('Reduction...')
    node_to_cut = list()
    for i_node in range(dTree.tree_.node_count):

        if ( dTree.tree_.feature[i_node] != -2 ):
            le = leaf_error(dTree.tree_,i_node )
            e,nn= error(dTree.tree_,i_node)
            if le <= e :
                node_to_cut.append(i_node)
    
    cut_into_leaves(dTree,node_to_cut)

    return dTree

iris = datasets.load_iris()
X = iris.data  
y = iris.target

K = 2
c = 0

#rf_s = list()
#ind_test = list()
#kf = StratifiedKFold(n_splits=K) 
#for train, test in kf.split(X,y):  
#    c = c+1      
#    #print(c)
#    rf_s.append(skl_ens.RandomForestClassifier(n_estimators=50,max_depth=10, oob_score=True))
#    rf_s[-1].fit(X[train],y[train])
#    ind_test.append(test)
#
#dTree = rf_s[0].estimators_[0]
X_target = X
y_target = y

train = np.linspace(0,y.size-1,y.size).astype(int)[::20]
all_ = np.linspace(0,y.size-1,y.size).astype(int)

rf = (skl_ens.RandomForestClassifier(n_estimators=50,max_depth=10, oob_score=True))
rf.fit(X[train],y[train])
dTree = rf.estimators_[0]

print('Arbre initial appr. sur '+str(train.size)+' données : '+str(dTree.tree_.node_count)+' noeuds')
#SER(d,X[ind_test[0]],y[ind_test[0]])

#==============================================================================
#       DEBUG TESTS
#==============================================================================

t_copy = copy.deepcopy(dTree.tree_)

sparseM = dTree.decision_path(X_target)
#target_values[:,:,0] = np.sum(sparseM.toarray()[np.where(y_target == 0)[0]],axis = 0)
#target_values[:,:,1] = np.sum(sparseM.toarray()[np.where(y_target == 1)[0]],axis = 0)

leaves = np.where(t_copy.feature == -2)[0]

#expansion
print('Expansion...')
stop = 0
for f in leaves :
    ind = np.where(sparseM.toarray()[:,f] == 1 )[0]
    
    if ind.size != 0 :
        Sv = X_target[ind]
        #build full new tree from f
        DT_to_add = sklearn.tree.DecisionTreeClassifier()
        DT_to_add.min_impurity_split = 0
        DT_to_add.fit(Sv,y_target[ind])
        fusionDecisionTree(dTree, f, DT_to_add)

t_copy = copy.deepcopy(dTree.tree_)
target_values = np.zeros((t_copy.node_count,1,t_copy.value.shape[-1]))
sparseM = dTree.decision_path(X_target)
for iy in range(target_values.shape[-1]):
    target_values[:,0,iy] = np.sum(sparseM.toarray()[np.where(y_target == iy)[0]],axis = 0)
   
updateValues(dTree.tree_,target_values)
print(str(dTree.tree_.node_count)+' noeuds apres expansion')

# TEST PREDICT CASSE
try:
    dTree.predict(X)
    print('PREDICT CONSERVE')
except:
    print('PREDICT CASSE ')

##reduction
print('Reduction...')
node_to_cut = list()
for i_node in range(dTree.tree_.node_count):
    if ( dTree.tree_.feature[i_node] != -2 ):
        le = leaf_error(dTree.tree_,i_node )
        e,nn= error(dTree.tree_,i_node)
        if le <= e :
            node_to_cut.append(i_node)

print('Indices à couper : ',node_to_cut)
cut_into_leaves(dTree,node_to_cut)

## TEST PREDICT CASSE
#try:
#    dTree.predict(X)
#    print('PREDICT CONSERVE')
#except:
#    print('PREDICT CASSE ')
#==============================================================================
# 
#==============================================================================


print('Arbre initial réajuté sur '+str(all_.size)+' données : '+str(dTree.tree_.node_count)+' noeuds')
