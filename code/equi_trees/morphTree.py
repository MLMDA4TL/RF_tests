#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 03:16:23 2018

@author: mounir
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz

from sklearn import datasets

def depth(d,node):
    if node == -1:
        return 0
    if node == 0:
        return 0
    else:
        return max(depth(d, d['nodes']['left_child'][node]), depth(d,d['nodes']['right_child'][node])) + 1
    
def all_thresh(dtree):
    nf = dtree.n_features_
    M = np.zeros(nf,dtype = object)
    sizes = np.zeros(nf,dtype = int)
    
    for i in range(nf):
        M[i] = list()
    d = dtree.tree_.__getstate__().copy()
    for n in d['nodes']:
        if ( n['feature'] != -2 ):
            M[n['feature']].append(n['threshold'])
            sizes[n['feature']] = sizes[n['feature']] + 1
        
    return M, sizes


def extract_rule(d,node):

    feats = list()
    ths = list()
    bools = list()
    nodes = list()
    b = 1
    if node != 0:
        while b != 0:
           
            feats.append(d['nodes']['feature'][node])
            ths.append(d['nodes']['threshold'][node])
            bools.append(b)
            nodes.append(node)
            node,b = find_parent(d,node)
            
            
        feats.pop(0)
        ths.pop(0)
        bools.pop(0)
        nodes.pop(0)
  
        
    return np.array(feats), np.array(ths), np.array(bools)
    
def extract_leaves_rules(d):
    leaves = np.where(d['nodes']['feature'] == -2)[0]
    
    rules = np.zeros(leaves.size,dtype = object)
    for k,f in enumerate(leaves) :
        rules[k] = extract_rule(d,f)
        
    return leaves, rules

#def is_in_subspace(dtree,feat,th,node):
#def paths_rule(dtree,rule):
#def leaf_center(dtree,node):

#def quadrillage(dtree):

def isdisj_feat(ths1,bools1,ths2,bools2):
    if np.sum(bools1 == -1) != 0:
        max_th1 = np.amin(ths1[bools1==-1])
    else:
        max_th1 = np.inf
        
    if np.sum(bools1 == 1) != 0:
        min_th1 = np.amax(ths1[bools1==1])
    else:
        min_th1 = - np.inf
    
    if np.sum(bools2 == -1) != 0:
        max_th2 = np.amin(ths2[bools2==-1])
    else: 
        max_th2 = np.inf
        
    if np.sum(bools2 == 1) != 0:
        min_th2 = np.amax(ths2[bools2==1])  
    else:
        min_th2 = - np.inf
    
    if ( min_th2> min_th1 and min_th2< max_th1 ) or ( max_th2> min_th1 and max_th2< max_th1 ) or ( max_th1> min_th2 and max_th1< max_th2 ) or ( min_th1> min_th2 and min_th1< max_th2 ) or ( min_th1 == min_th2 and max_th1 == max_th2 )   :
        return 0
    else:
        return 1
    
def isdisj(rule1,rule2):
    feats1, ths1, bools1 = rule1
    feats2, ths2, bools2 = rule2
    if np.array(rule1).size == 0 or np.array(rule2).size == 0 :
        return 0
    isdj = 0

    for phi in feats1:
        
        if phi in feats2:
            
            ths1_f = ths1[ feats1 == phi ]
            ths2_f = ths2[ feats2 == phi ]
            bools1_f = bools1[ feats1 == phi ]
            bools2_f = bools2[ feats2 == phi ]
            
            if isdisj_feat(ths1_f,bools1_f,ths2_f,bools2_f):
                isdj = 1

    
    return isdj
    
    
def reaching_class(d,rule):
    class_list = list()
    leaves, rules = extract_leaves_rules(d)
    for k,l in enumerate(leaves):
        c = int(np.argmax(d['values'][l,:,:]))
        
        if c not in class_list:
            if not isdisj(rules[k],rule):
                class_list.append(c)
                
    return class_list


       
def coherent_new_split(dic_or,phi,th,rule):
    coherent_regardless_class = 0
    still_splitting = 0
    
    feats, ths, bools = rule
    
    if phi not in feats:
        coherent_regardless_class = 1
    else:
        if np.sum((feats == phi)*(bools==-1)) != 0:
            max_th = np.amin(ths[(feats == phi)*(bools==-1)])
        else:
            max_th = np.inf
        
        if np.sum((feats == phi)*(bools==1)) != 0:
            min_th = np.amax(ths[(feats == phi)*(bools==1)])
        else:
            min_th = - np.inf
        
        if th > max_th or th < min_th:
            coherent_regardless_class = 0
        else:
            coherent_regardless_class = 1
        
        
    new_f = np.concatenate((feats,np.array([phi])))
    new_t = np.concatenate((ths,np.array([th])))
    new_bl = np.concatenate((bools,np.array([-1])))
    new_br = np.concatenate((bools,np.array([1])))
    
    
    if len(reaching_class(dic_or,(new_f,new_t,new_bl))) > 0 and len(reaching_class(dic_or,(new_f,new_t,new_br))) > 0:
        still_splitting = 1

    return still_splitting*coherent_regardless_class
        
    
def isinrule(rule, split):
    f,t = split
    
    feats, ths, bools = rule
    for k,f2 in enumerate(feats):

        if f2 == f and t == ths[k]:
            return 1
    return 0

def new_split(rule,all_splits):
    if np.array(rule).size > 0 :
        splits_not_in_rule = np.zeros(all_splits.size)

        for i,split in enumerate(all_splits):
            splits_not_in_rule[i] = 1 - isinrule(rule,split)

        ind = np.random.randint(all_splits[splits_not_in_rule.astype(bool)].size)
        ind = np.linspace(0,all_splits.size-1,all_splits.size).astype(int)[splits_not_in_rule.astype(bool)][ind]
    else:
        ind = np.random.randint(all_splits.size)
    return ind

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

def add_to_parents(d, node, values):

    p,b = find_parent(d,node)
    if b != 0: 
        d['values'][p] =  d['values'][p] + values
        add_to_parents(d, p, values)
        
        
def add_child_leaf(d,node,lr):
    new_node = np.zeros(1,dtype=[('left_child', '<i8'),
                                 ('right_child', '<i8'), ('feature', '<i8'),
                                 ('threshold', '<f8'), ('impurity', '<f8'),
                                 ('n_node_samples', '<i8'), ('weighted_n_node_samples', '<f8')])
    
    new_node[0]['left_child'] = -1
    new_node[0]['right_child'] = -1
    new_node[0]['feature'] = -2
    new_node[0]['threshold'] = -1
    #new_node[0]['impurity'] = 
    new_node[0]['n_node_samples'] = 0
    new_node[0]['weighted_n_node_samples'] = 0
    
    d['nodes'] = np.concatenate((d['nodes'],new_node))
    d['values'] = np.concatenate(( d['values'],np.zeros((1,d['values'].shape[1],d['values'].shape[2]))))
    
    if lr == -1:
        d['nodes']['left_child'][node] = d['nodes'].size - 1
    if lr == 1:
        d['nodes']['right_child'][node] = d['nodes'].size - 1
        
    d['node_count'] = d['node_count'] + 1
    
    dep = depth(d,node)
    if ( dep > d['max_depth'] ):
        d['max_depth'] = dep
        
    return d, d['nodes'].size - 1
    
def rec_split(dic_or,node,d,all_splits):
    print('Node',node)
    #print(d)
    rule = extract_rule(d,node)
    #print('Rule :',rule)
    cs = reaching_class(dic_or,rule)
    print('reaching classes',cs)
    if len(cs) > 1 :
        print('New split')
        is_coherent = 0
        while not is_coherent:
            ind = new_split(rule,all_splits)
            phi, th = all_splits[ind]
            is_coherent = coherent_new_split(dic_or,phi,th,rule)

        d['nodes']['feature'][node] = phi
        d['nodes']['threshold'][node] = th
        #Create children first as leaves
        d, child_l = add_child_leaf(d,node,-1)
        d = rec_split(dic_or,child_l,d,all_splits)
        
        #print('passage noeud droit')
        d, child_r = add_child_leaf(d,node,1)
        d = rec_split(dic_or,child_r,d,all_splits)
            
    elif len(cs) == 1:
        print('New leaf')
        #pour l'instant...
        c = cs[0]
        
        ###
        d['values'][node,:,c] = 1
        add_to_parents(d, node, d['values'][node])
        
        d['nodes']['n_node_samples'][node] = 1
        d['nodes']['weighted_n_node_samples'][node] = 1
    
    else:
        print('0 données !')

    return d
    
def equivalent_random(dtree):
    dic_or = dtree.tree_.__getstate__().copy()
    
    leaves = np.where(dtree.tree_.feature == -2)[0] 
    all_splits = np.zeros(dtree.tree_.node_count - leaves.size,dtype=object)
    
    compt = 0
    for i in range(dtree.tree_.node_count):
        if i not in leaves:
            all_splits[compt]= (dtree.tree_.feature[i],dtree.tree_.threshold[i])
            compt = compt + 1
    print(all_splits)

    d = dict()
    d['node_count'] = 1
    d['max_depth'] = 0 
    d['nodes'] = np.zeros(1,dtype=[('left_child', '<i8'),
                                 ('right_child', '<i8'), ('feature', '<i8'),
                                 ('threshold', '<f8'), ('impurity', '<f8'),
                                 ('n_node_samples', '<i8'), ('weighted_n_node_samples', '<f8')])
    d['values'] = np.zeros((1,dic_or['values'].shape[1],dic_or['values'].shape[2]))
    
    #initial node
    k = np.random.randint(all_splits.size)
    phi_init, th_init = all_splits[k]
    d['nodes'][0]['feature'] = phi_init
    d['nodes'][0]['threshold'] = th_init
    d['nodes'][0]['left_child'] = -1
    d['nodes'][0]['right_child'] = -1
  
    d = rec_split(dic_or,0,d,all_splits)
    print('nb noeuds :',d['node_count'])
    (Tree,(n_f,n_c,n_o),b) = dtree.tree_.__reduce__()
    new_tree = Tree(n_f, n_c, n_o)

    new_tree.__setstate__(d)  
    
    print('nb noeuds :',new_tree.__getstate__()['nodes'].size)
    
    new_dtree = DecisionTreeClassifier()
    new_dtree.n_features_ = n_f
    new_dtree.n_classes_ = n_c[0]
    #print(n_c)
    new_dtree.classes_ = np.linspace(0,n_c[0]-1,n_c[0]).astype(int)
    new_dtree.n_outputs_ = n_o
    new_dtree.tree_ = new_tree
    new_dtree.max_depth = new_tree.max_depth
    
    return new_dtree
    
if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    dtree = DecisionTreeClassifier()
    dtree.fit(X,y)
    export_graphviz(dtree, "my_output/dtree.dot")
    new_dtree = equivalent_random(dtree)
    export_graphviz(new_dtree, "my_output/new_dtree.dot")
