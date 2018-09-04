#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 15:43:47 2018

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

from sklearn.tree import export_graphviz

#==============================================================================
#
#==============================================================================


def updateValues(tree, values):
    d = tree.__getstate__()
    d['values'] = values
    d['nodes']['n_node_samples'] = np.sum(values, axis=-1).reshape(-1)
    d['nodes']['weighted_n_node_samples'] = np.sum(values, axis=-1).reshape(-1)
    tree.__setstate__(d)


def depth(tree, f):
    if f == -1:
        return 0
    if f == 0:
        return 0
    else:
        return max(depth(tree, tree.children_left[f]), depth(tree, tree.children_right[f])) + 1


def depth_array(tree, inds):
    depths = np.zeros(np.array(inds).size)
    for i, e in enumerate(inds):
        depths[i] = depth(tree, i)
    return depths


def max_depth_dTree(dTree):
    t = dTree.tree_
    n = t.node_count

    return np.amax(depth_array(t, np.linspace(0, n - 1, n).astype(int)))


def max_depth_rf(rf):
    p = 0
    for e in rf.estimators_:
        if max_depth_dTree(e) > p:
            p = max_depth_dTree(e)
    return p


def leaf_error(tree, node):
    if np.sum(tree.value[node]) == 0:
        return 0
    else:
        return 1 - np.max(tree.value[node]) / np.sum(tree.value[node])


def error(tree, node):
    if node == -1:
        return 0
    else:

        if tree.feature[node] == -2:
            return leaf_error(tree, node)
        else:
            # Pas une feuille

            nr = np.sum(tree.value[tree.children_right[node]])
            nl = np.sum(tree.value[tree.children_left[node]])

            if nr + nl == 0:
                return 0
            else:
                er = error(tree, tree.children_right[node])
                el = error(tree, tree.children_left[node])

                return (el * nl + er * nr) / (nl + nr)


def find_parent(dic, i_node):
    p = -1
    b = 0
    if i_node != 0 and i_node != -1:

        try:
            p = list(dic['nodes']['left_child']).index(i_node)
            b = -1
        except:
            p = p
        try:
            p = list(dic['nodes']['right_child']).index(i_node)
            b = 1
        except:
            p = p

    return p, b


def sub_nodes(tree, node):
    if (node == -1):
        return list()
    if (tree.feature[node] == -2):
        return [node]
    else:
        return [node] + sub_nodes(tree, tree.children_left[node]) + sub_nodes(tree, tree.children_right[node])


def fusionTree(tree1, f, tree2):
    """adding tree tree2 to leaf f of tree tree1"""

    dic = tree1.__getstate__().copy()
    dic2 = tree2.__getstate__().copy()

    size_init = tree1.node_count

    if depth(tree1, f) + dic2['max_depth'] > dic['max_depth']:
        dic['max_depth'] = depth(tree1, f) + tree2.max_depth

    dic['capacity'] = tree1.capacity + tree2.capacity - 1
    dic['node_count'] = tree1.node_count + tree2.node_count - 1

    dic['nodes'][f] = dic2['nodes'][0]

    if (dic2['nodes']['left_child'][0] != - 1):
        dic['nodes']['left_child'][f] = dic2[
            'nodes']['left_child'][0] + size_init - 1
    else:
        dic['nodes']['left_child'][f] = -1
    if (dic2['nodes']['right_child'][0] != - 1):
        dic['nodes']['right_child'][f] = dic2[
            'nodes']['right_child'][0] + size_init - 1
    else:
        dic['nodes']['right_child'][f] = -1

    # Attention vecteur impurity pas mis à jour

    dic['nodes'] = np.concatenate((dic['nodes'], dic2['nodes'][1:]))
    dic['nodes']['left_child'][size_init:] = (dic['nodes']['left_child'][
                                              size_init:] != -1) * (dic['nodes']['left_child'][size_init:] + size_init) - 1
    dic['nodes']['right_child'][size_init:] = (dic['nodes']['right_child'][
                                               size_init:] != -1) * (dic['nodes']['right_child'][size_init:] + size_init) - 1

    values = np.concatenate((dic['values'], np.zeros((dic2['values'].shape[
                            0] - 1, dic['values'].shape[1], dic['values'].shape[2]))), axis=0)

    dic['values'] = values

    # Attention :: (potentiellement important)
    (Tree, (n_f, n_c, n_o), b) = tree1.__reduce__()
    #del tree1
    #del tree2

    tree1 = Tree(n_f, n_c, n_o)

    tree1.__setstate__(dic)
    return tree1


def fusionDecisionTree(dTree1, f, dTree2):
    """adding tree dTree2 to leaf f of tree dTree1"""
    #dTree = sklearn.tree.DecisionTreeClassifier()
    size_init = dTree1.tree_.node_count
    dTree1.tree_ = fusionTree(dTree1.tree_, f, dTree2.tree_)

    try:
        dTree1.tree_.value[size_init:, :, dTree2.classes_.astype(
            int)] = dTree2.tree_.value[1:, :, :]
    except IndexError as e:
        print("IndexError : size init : ", size_init,
              "\ndTree2.classes_ : ", dTree2.classes_)
        print(e)
    dTree1.max_depth = dTree1.tree_.max_depth
    return dTree1

# def exchange_nodes(dTree,node1,node2):
#    if node1 == node2 :
#        return 0
#
#    tree = dTree.tree_
#    dic = tree.__getstate__().copy()
#    inter = ( dic['nodes'][node1] , dic['values'][node1] )
#
#    dic['nodes'][node1] = dic['nodes'][node2]
#    dic['values'][node1] = dic['values'][node2]
#
#    nodes, val = inter
#    dic['nodes'][node2] = nodes
#    dic['values'][node2] = val
#
#    tree.__setstate__(dic)
#
#    return 1
#


#
def cut_from_left_right(dTree, node, bool_left_right):
    dic = dTree.tree_.__getstate__().copy()
    #dic_o = tree.__getstate__()
    #dic= dic_o.copy()

    node_to_rem = list()
    size_init = dTree.tree_.node_count

    p, b = find_parent(dic, node)

    if bool_left_right == 1:
        repl_node = dTree.tree_.children_left[node]
        #node_to_rem = [node] + sub_nodes(dTree.tree_,dTree.tree_.children_right[node])
        node_to_rem = [node, dTree.tree_.children_right[node]]
    elif bool_left_right == -1:
        repl_node = dTree.tree_.children_right[node]
        #node_to_rem = [node] + sub_nodes(dTree.tree_,dTree.tree_.children_left[node])
        node_to_rem = [node, dTree.tree_.children_left[node]]

    inds = list(
        set(np.linspace(0, size_init - 1, size_init).astype(int)) - set(node_to_rem))

    dic['capacity'] = dTree.tree_.capacity - len(node_to_rem)
    dic['node_count'] = dTree.tree_.node_count - len(node_to_rem)

    if b == 1:
        dic['nodes']['right_child'][p] = repl_node
    elif b == -1:
        dic['nodes']['left_child'][p] = repl_node

    #new_size = len(ind)
    dic_old = dic.copy()
    left_old = dic_old['nodes']['left_child']
    right_old = dic_old['nodes']['right_child']
    #print('taille avant:',dic['nodes'].size)
    dic['nodes'] = dic['nodes'][inds]
    dic['values'] = dic['values'][inds]
    #print('taille après:',dic['nodes'].size)

    for i, new in enumerate(inds):
        if (left_old[new] != -1):
            dic['nodes']['left_child'][i] = inds.index(left_old[new])
        else:
            dic['nodes']['left_child'][i] = -1
        if (right_old[new] != -1):
            dic['nodes']['right_child'][i] = inds.index(right_old[new])
        else:
            dic['nodes']['right_child'][i] = -1

    (Tree, (n_f, n_c, n_o), b) = dTree.tree_.__reduce__()
    del dTree.tree_

    dTree.tree_ = Tree(n_f, n_c, n_o)
    dTree.tree_.__setstate__(dic)
    depths = depth_array(dTree.tree_, np.linspace(
        0, dTree.tree_.node_count - 1, dTree.tree_.node_count).astype(int))
    dTree.tree_.max_depth = np.max(depths)

    return inds.index(repl_node)


def cut_into_leaf2(dTree, node):
    dic = dTree.tree_.__getstate__().copy()

    node_to_rem = list()
    size_init = dTree.tree_.node_count

    node_to_rem = node_to_rem + sub_nodes(dTree.tree_, node)[1:]
    node_to_rem = list(set(node_to_rem))

    inds = list(
        set(np.linspace(0, size_init - 1, size_init).astype(int)) - set(node_to_rem))
    depths = depth_array(dTree.tree_, inds)
    dic['max_depth'] = np.max(depths)

    dic['capacity'] = dTree.tree_.capacity - len(node_to_rem)
    dic['node_count'] = dTree.tree_.node_count - len(node_to_rem)

    dic['nodes']['feature'][node] = -2
    dic['nodes']['left_child'][node] = -1
    dic['nodes']['right_child'][node] = -1

    #new_size = len(ind)
    dic_old = dic.copy()
    left_old = dic_old['nodes']['left_child']
    right_old = dic_old['nodes']['right_child']
    #print('taille avant:',dic['nodes'].size)
    dic['nodes'] = dic['nodes'][inds]
    dic['values'] = dic['values'][inds]
    #print('taille après:',dic['nodes'].size)

    for i, new in enumerate(inds):
        if (left_old[new] != -1):
            dic['nodes']['left_child'][i] = inds.index(left_old[new])
        else:
            dic['nodes']['left_child'][i] = -1
        if (right_old[new] != -1):
            dic['nodes']['right_child'][i] = inds.index(right_old[new])
        else:
            dic['nodes']['right_child'][i] = -1

    (Tree, (n_f, n_c, n_o), b) = dTree.tree_.__reduce__()
    del dTree.tree_

    dTree.tree_ = Tree(n_f, n_c, n_o)
    dTree.tree_.__setstate__(dic)

    return inds.index(node)


def SER(node, dTree, X_target_node, y_target_node, no_red_on_cl=False, cl_no_red=None, no_ser_on_cl=False, cl_no_ser=None):

    # Maj values
    #old_val = dTree.tree_.value[node]
    val = np.zeros((dTree.n_outputs_, dTree.n_classes_))

    for i in range(dTree.n_classes_):
        val[:, i] = list(y_target_node).count(i)

    old_size_cl_no_red = np.sum(dTree.tree_.value[node][:, cl_no_red])

    if no_red_on_cl and dTree.tree_.feature[node] == -2 and y_target_node.size == 0 and old_size_cl_no_red > 0:
        v = np.zeros((dTree.n_outputs_, dTree.n_classes_))
        val[:, cl_no_red] = dTree.tree_.value[node][:, cl_no_red]

        v[:, cl_no_red] = val[:, cl_no_red]
        add_to_parents(dTree, node, v)

    dTree.tree_.value[node] = val
    dTree.tree_.n_node_samples[node] = np.sum(val)
    dTree.tree_.weighted_n_node_samples[node] = np.sum(val)

    ### EXPANSION ###

    # Si c'est une feuille
    if dTree.tree_.feature[node] == -2:
        if no_ser_on_cl:
            if np.sum(dTree.tree_.value[node, :]) > 0 and np.argmax(dTree.tree_.value[node, :]) not in cl_no_ser:
                DT_to_add = sklearn.tree.DecisionTreeClassifier()
                # to make a complete tree
                try:
                    DT_to_add.min_impurity_decrease = 0
                except:
                    DT_to_add.min_impurity_split = 0
                DT_to_add.fit(X_target_node, y_target_node)
                fusionDecisionTree(dTree, node, DT_to_add)
            # else:
                # print('Feuille laissée intacte')
#
        else:
            # Si elle n'est pas déjà pure
            if (len(set(list(y_target_node))) > 1):
                # build full new tree from f
                DT_to_add = sklearn.tree.DecisionTreeClassifier()
                # to make a complete tree
                try:
                    DT_to_add.min_impurity_decrease = 0
                except:
                    DT_to_add.min_impurity_split = 0
                DT_to_add.fit(X_target_node, y_target_node)
                fusionDecisionTree(dTree, node, DT_to_add)

        return node

    # Si ce n'est pas une feuille

    bool_test = X_target_node[:, dTree.tree_.feature[
        node]] <= dTree.tree_.threshold[node]
    not_bool_test = X_target_node[
        :, dTree.tree_.feature[node]] > dTree.tree_.threshold[node]

    ind_left = np.where(bool_test)[0]
    ind_right = np.where(not_bool_test)[0]

    X_target_node_left = X_target_node[ind_left]
    y_target_node_left = y_target_node[ind_left]

    X_target_node_right = X_target_node[ind_right]
    y_target_node_right = y_target_node[ind_right]

    #<----- CHGT STRUCTURE :"node" DOIT CHANGER ( OK REC )
    new_node_left = SER(dTree.tree_.children_left[node], dTree, X_target_node_left, y_target_node_left,
                        no_red_on_cl=no_red_on_cl, cl_no_red=cl_no_red,
                        no_ser_on_cl=no_ser_on_cl, cl_no_ser=cl_no_ser)
    dic = dTree.tree_.__getstate__().copy()
    node, b = find_parent(dic, new_node_left)
    new_node_right = SER(dTree.tree_.children_right[node], dTree, X_target_node_right, y_target_node_right,
                         no_red_on_cl=no_red_on_cl, cl_no_red=cl_no_red,
                         no_ser_on_cl=no_ser_on_cl, cl_no_ser=cl_no_ser)
    dic = dTree.tree_.__getstate__().copy()
    node, b = find_parent(dic, new_node_right)

    #export_graphviz(dTree, "output/dtree_after_exp_test.dot")
    ### REDUCTION ###
    le = leaf_error(dTree.tree_, node)
    e = error(dTree.tree_, node)

    if le <= e:

        #<----- CHGT STRUCTURE :"node" DOIT CHANGER ( OK )
        new_node_leaf = cut_into_leaf2(dTree, node)
        node = new_node_leaf

    if dTree.tree_.feature[node] != -2:
        # Normalement, on passe sur un noeud atteint ( donc les 2 pas zero
        # simult.)
        if no_red_on_cl:
            if ind_left.size == 0 and np.sum(dTree.tree_.value[dTree.tree_.children_left[node]]) == 0:
                node = cut_from_left_right(dTree, node, -1)

            if ind_right.size == 0 and np.sum(dTree.tree_.value[dTree.tree_.children_right[node]]) == 0:
                node = cut_from_left_right(dTree, node, 1)
        else:
            if ind_left.size == 0:
                node = cut_from_left_right(dTree, node, -1)

            if ind_right.size == 0:
                node = cut_from_left_right(dTree, node, 1)

    #export_graphviz(dTree, "output/dtree_after_red_test.dot")

    # On en a besoin pour mettre à jour les vecteurs n_node_samples & weighted_n_node_samples à partir des values
    # Peut être à revoir plus tard
#    if node == 0:
#        updateValues(dTree.tree_, dTree.tree_.value)

    return node


def add_to_parents(dTree, node, values):
    dic = dTree.tree_.__getstate__().copy()
    p, b = find_parent(dic, node)
    if b != 0:
        dTree.tree_.value[p] = dTree.tree_.value[p] + values
        add_to_parents(dTree, p, values)


def add_to_child(dTree, node, values):

    l = dTree.tree_.children_left[node]
    r = dTree.tree_.children_right[node]

    if r != -1:
        dTree.tree_.value[r] = dTree.tree_.value[r] + values
        add_to_child(dTree, r, values)
    if l != -1:
        dTree.tree_.value[l] = dTree.tree_.value[l] + values
        add_to_child(dTree, l, values)


def bootstrap(size):
    return np.random.choice(np.linspace(0, size - 1, size).astype(int), size, replace=True)


def SER_RF(random_forest, X_target, y_target, bootstrap_=False, no_red_on_cl=False, cl_no_red=None, no_ser_on_cl=False, cl_no_ser=None):
    rf_ser = copy.deepcopy(random_forest)
    for i, dtree in enumerate(rf_ser.estimators_):
        # print("tree n° ", i)

        inds = np.linspace(0, y_target.size - 1, y_target.size).astype(int)
        if bootstrap_:
            inds = bootstrap(y_target.size)

        SER(0, rf_ser.estimators_[i], X_target[inds], y_target[inds],
            no_red_on_cl=no_red_on_cl, cl_no_red=cl_no_red,
            no_ser_on_cl=no_ser_on_cl, cl_no_ser=cl_no_ser)
    return rf_ser
