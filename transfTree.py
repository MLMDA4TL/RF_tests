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
import copy
import STRUT as strut
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
    """adding tree tree2 to leaf f of tree tree1"""

    dic = tree1.__getstate__().copy()
    dic2 = tree2.__getstate__()
    
    size_init = tree1.node_count

    if depth(tree1, f) + dic2['max_depth'] > dic['max_depth']:
        dic['max_depth'] = depth(tree1, f) + tree2.max_depth

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
    """adding tree dTree2 to leaf f of tree dTree1"""

    dTree1.tree_ = fusionTree(dTree1.tree_, f, dTree2.tree_)
    dTree1.max_depth = dTree1.tree_.max_depth
    return dTree1

  
    
def leaf_error(tree,node):
#    if node == -1 :
#        return 0
#    else:
    if np.sum(tree.value[node]) == 0 :
        return 0
    else:
        return 1 - np.max(tree.value[node])/np.sum(tree.value[node])

def error(tree,node):
    if node == -1 : 
        return 0,0
    else: 
    
        if tree.feature[node] == -2 :
            return leaf_error(tree,node),1
        else:
            er, nr = error(tree, tree.children_right[node])
            el, nl = error(tree, tree.children_left[node])
            return (el * nl + er * nr) / (nl + nr), nl + nr


def sub_nodes(tree, node):
    if (node == -1):
        return list()
    if (tree.feature[node] == -2):
        return [node]
    else:
        return [node] + sub_nodes(tree, tree.children_left[node]) + sub_nodes(tree, tree.children_right[node])


def cut_into_leaves_tree(tree, nodes):
    dic = tree.__getstate__().copy()
    node_to_rem = list()
    nodes = list(set(nodes))
    
    size_init = tree.node_count
    
    for node in nodes:
        
        node_to_rem = node_to_rem + sub_nodes(tree,node)[1:]
        
    node_to_rem = list(set(node_to_rem))
    
    depths = depth_array(tree,list(set(np.linspace(0,size_init-1,size_init).astype(int))-set(node_to_rem)))
    dic['max_depth'] = np.max(depths)

    dic['capacity'] = tree.capacity - len(node_to_rem)
    dic['node_count'] = tree.node_count - len(node_to_rem)

    dic['nodes']['feature'][nodes] = -2
    dic['nodes']['left_child'][nodes] = -1
    dic['nodes']['right_child'][nodes] = -1
    
    ind = list(set(np.linspace(0,size_init-1,size_init).astype(int))-set(node_to_rem))
                
    dic['nodes'] = dic['nodes'][ind]
    dic['values'] = dic['values'][ind]

    for i in range(dic['nodes']['left_child'].size):
        if ( dic['nodes']['left_child'][i] != -1 ):        
            dic['nodes']['left_child'][i] = ind.index(dic['nodes']['left_child'][i])

    for i in range(dic['nodes']['right_child'].size):
        if ( dic['nodes']['right_child'][i] != -1 ):        
            dic['nodes']['right_child'][i] = ind.index(dic['nodes']['right_child'][i])
            
    (Tree,(n_f,n_c,n_o),b) = tree.__reduce__()
    del tree

    tree = Tree(n_f, n_c, n_o)
    tree.__setstate__(dic)

    return tree


def cut_into_leaves(dTree, nodes):
    dTree.tree_ = cut_into_leaves_tree(dTree.tree_, nodes)
    dTree.max_depth = dTree.tree_.max_depth


def SER(dTree, X_target, y_target):

    t_copy = copy.deepcopy(dTree.tree_)
    sparseM = dTree.decision_path(X_target)
    leaves = np.where(t_copy.feature == -2)[0]

    # expansion
    print('Expansion...')
    for f in leaves:
        # indices of instances ending up in leaf f
        ind = np.where(sparseM.toarray()[:, f] == 1)[0]

        if ind.size != 0:
            Sv = X_target[ind]

            # build full new tree from f
            DT_to_add = sklearn.tree.DecisionTreeClassifier()
            # to make a complete tree
            DT_to_add.min_impurity_split = 0
            DT_to_add.fit(Sv, y_target[ind])
            fusionDecisionTree(dTree, f, DT_to_add)

    t_copy = copy.deepcopy(dTree.tree_)
    target_values = np.zeros((t_copy.node_count, 1, t_copy.value.shape[-1]))
    sparseM = dTree.decision_path(X_target)

    for iy in range(target_values.shape[-1]):
        target_values[:, 0, iy] = np.sum(
            sparseM.toarray()[np.where(y_target == iy)[0]], axis=0)

    updateValues(dTree.tree_, target_values)
    print(str(dTree.tree_.node_count) + ' noeuds apres expansion')

    # reduction
    print('Reduction...')
    node_to_cut = list()
    for i_node in range(dTree.tree_.node_count):
        if (dTree.tree_.feature[i_node] != -2):
            le = leaf_error(dTree.tree_, i_node)
            e, nn = error(dTree.tree_, i_node)
            if le <= e:
                node_to_cut.append(i_node)

    cut_into_leaves(dTree, node_to_cut)

    print(str(dTree.tree_.node_count) + ' noeuds apres reduction')
    return dTree


def SER_RF(random_forest, X_target, y_target):
    rf_ser = copy.deepcopy(random_forest)
    for i, dtree in enumerate(rf_ser.estimators_):
        rf_ser.estimators_[i] = SER(rf_ser.estimators_[i], X_target, y_target)
    return rf_ser

def all_splits(X,ind_feats,min_delta = 10e-7):
    x_sorted = X[:,ind_feats].copy()
    th = np.zeros(ind_feats.size,dtype = object)
    inds = np.zeros(ind_feats.size,dtype = object) 
    
    for c in range(x_sorted.shape[1]):
        x_sorted[:,c] = np.sort(x_sorted[:,c])
        
        var = x_sorted[1:,c]- x_sorted[:-1,c]
        inds[c] = np.where(var > min_delta*(max(x_sorted[:,c]) - min(x_sorted[:,c])))[0]  
        th[c] = (x_sorted[inds[c]+1,c]+x_sorted[inds[c],c] ) /2
        
    return th

def search_best_split(decision_tree,node_index, ind_feats, Q_source_parent,
                        X_target_node,
                        Y_target_node,
                        classes, measure_func = [[strut.DG,strut.IG]]):
    
    ths = all_splits(X,ind_feats,min_delta = 10e-7)

    #Q_source_l,Q_source_r = get_children_distributions(decisiontree,node_index)
    #Q_source_parent = get_node_distribution(decisiontree,node_index)
        
    bool_ = 0
    f_opt = -1
    #ind_th_opt = -1
    
    ind_f = -1
    #ind_th = -1
    
    and_block_score = np.zeros(len(measure_func),dtype = object)
    and_block_score_opt = np.zeros(len(measure_func),dtype = object)
    for i,and_block in enumerate(measure_func):
        and_block_score[i] = np.zeros(len(and_block))
        and_block_score_opt[i] = np.zeros(len(and_block))
        
    for f in ths:
        ind_f =  ind_f + 1
        for th in f:
            #ind_th = ind_th + 1
            
            
            Q_target_l,Q_target_r = strut.compute_Q_children_target(X_target_node,
                                                              Y_target_node,
                                                              f,
                                                              th,
                                                              classes)
            for i,and_block in enumerate(measure_func):
                bool_and = 1
                for j,func in enumerate(and_block) :
                    and_block_score[i][j] = func( Q_source_parent,[Q_target_l,Q_target_r])
                    if and_block_score[i][j] <= and_block_score_opt[i][j] :
                        bool_and = 0
                    else:
                        and_block_score_opt[i][j] = and_block_score[i][j]
                        
                bool_ = bool_ + bool_and
                
            if (bool_):
                f_opt = ind_f
                #ind_th_opt = ind_th
                th_opt = th
    
    return  ind_feats[f_opt], th_opt

if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    K = 2
    c = 0

    X_target = X
    y_target = y

    train = np.linspace(0,y.size-1,y.size).astype(int)[::20]
    all_ = np.linspace(0,y.size-1,y.size).astype(int)
    test = list(set(all_) - set(train))

    rf = (skl_ens.RandomForestClassifier(n_estimators=50,max_depth=10, oob_score=True))
    rf.fit(X[train],y[train])
    dTree = rf.estimators_[0]

    print('Arbre initial appr. sur '+str(train.size)+' données : '+str(dTree.tree_.node_count)+' noeuds')
    SER(dTree,X[test],y[test])


    print('Arbre initial réajuté sur '+str(all_.size)+' données : '+str(dTree.tree_.node_count)+' noeuds')

