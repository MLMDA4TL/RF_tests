# -*- coding: utf-8 -*-
"""
@author: sergio
"""

import numpy as np
from sklearn import tree

def get_children_distributions(decisiontree,
                               node_index):
    tree = decisiontree.tree_
    child_l = tree.children_left[node_index]
    child_r = tree.children_right[node_index]
    Q_source_l = tree.value[child_l]
    Q_source_r = tree.value[child_r]
    return [np.asarray(Q_source_l), np.asarray(Q_source_r)]
                                              
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

def compute_Q_children_target(X_target_node,
                              Y_target_node,
                              phi,
                              threshold,
                              classes):
    # Split parent node target sample using the threshold provided
    # instances <= threshold go to the left 
    # instances > threshold go to the right 
    decision_l = X_target_node[:,phi] <= threshold
    decision_r = np.logical_not(decision_l)
    Y_target_child_l = Y_target_node[decision_l]
    Y_target_child_r = Y_target_node[decision_r]
    Q_target_l = compute_class_distribution(classes, Y_target_child_l)
    Q_target_r = compute_class_distribution(classes, Y_target_child_r) 
    return Q_target_l,Q_target_r


def threshold_selection(Q_source_parent,
                        Q_source_l,
                        Q_source_r,
                        X_target_node,
                        Y_target_node,
                        phi,
                        classes):
    # sort the corrdinates of X along phi
    X_phi_sorted = np.sort(X_target_node[:,phi])
    nb_tested_thresholds = X_target_node.shape[0] - 1
    measures_IG = np.zeros(nb_tested_thresholds)
    measures_DG = np.zeros(nb_tested_thresholds)
    for i in range(nb_tested_thresholds):
        threshold = (X_phi_sorted[i] + X_phi_sorted[i+1]) * 1./2
        Q_target_l,Q_target_r = compute_Q_children_target(X_target_node,
                                                          Y_target_node,
                                                          phi,
                                                          threshold,
                                                          classes)
        
        
        measures_IG[i] = IG(Q_source_parent,
                            [Q_target_l,Q_target_r])
        measures_DG[i] = DG(Q_source_l,
                            Q_source_r,
                            Q_target_l,
                            Q_target_r)
    index = 0
    for i in range(1,nb_tested_thresholds-1):
        if measures_IG[i] >= measures_IG[i-1] and measures_IG[i] >= measures_IG[i+1] and measures_DG[i] > measures_DG[index]:
            index = i
    threshold = (X_phi_sorted[index] + X_phi_sorted[index+1]) * 1./2
    return threshold

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
    p = class_distribution / class_distribution.sum()
    return 1 - (p**2).sum()
        
def STRUT(decisiontree,
          node_index,
          X_target_node,
          Y_target_node):
    tree = decisiontree.tree_
    phi = tree.feature[node_index]
    classes = decisiontree.classes_
    threshold = tree.threshold[node_index]
    current_class_distribution = compute_class_distribution(classes,Y_target_node)
    tree.weighted_n_node_samples[node_index] = Y_target_node.size
    tree.value[node_index] = current_class_distribution
    tree.impurity[node_index] = GINI(current_class_distribution)
    tree.n_node_samples[node_index] = Y_target_node.size
    
    # Unreachable node
    if current_class_distribution.sum() == 0 :
        prune_subtree(decisiontree,
                      node_index)
        tree.feature[node_index] = -2
        tree.children_left[tree.children_left == node_index] = -1
        tree.children_right[tree.children_right == node_index] = -1
        return 0

    # If it is a leaf one, exit
    if tree.children_left[node_index] == -1 and tree.children_right[node_index] == -1:
        return 0
    
    # Only one class is present in the node -> terminal leaf
    if (current_class_distribution > 0).sum() == 1 :
        prune_subtree(decisiontree,
                      node_index)
        tree.feature[node_index] = -2
        if Y_target_node.size >0:
            tree.value[node_index] = np.asarray([compute_class_distribution(classes,Y_target_node)])
        return 0
    
    # update threshold
    if type(threshold) is np.float64:
        Q_source_l,Q_source_r = get_children_distributions(decisiontree,
                                                           node_index)
        Q_source_parent = get_node_distribution(decisiontree,
                                                node_index)
        t1 = threshold_selection(Q_source_parent,
                                 Q_source_l,
                                 Q_source_r,
                                 X_target_node,
                                 Y_target_node,
                                 phi,
                                 classes)
        Q_target_l,Q_target_r = compute_Q_children_target(X_target_node,
                                                          Y_target_node,
                                                          phi,
                                                          t1,
                                                          classes)
        DG_t1 = DG(Q_source_l,
                   Q_source_r,
                   Q_target_l,
                   Q_target_r)
        t2 = threshold_selection(Q_source_parent,
                                 Q_source_r,
                                 Q_source_l,
                                 X_target_node,
                                 Y_target_node,
                                 phi,
                                 classes)
        Q_target_l,Q_target_r = compute_Q_children_target(X_target_node,
                                                          Y_target_node,
                                                          phi,
                                                          t2,
                                                          classes)
        DG_t2 = DG(Q_source_r,
                   Q_source_l,
                   Q_target_l,
                   Q_target_r)
        if DG_t1 >= DG_t2:
            tree.threshold[node_index] = t1
        else:
            tree.threshold[node_index] = t2
            # swap children
            old_child_r_id = tree.children_right[node_index]
            tree.children_right[node_index] = tree.children_left[node_index]
            tree.children_left[node_index] = old_child_r_id
    
    if tree.children_left[node_index] != -1:
        threshold = tree.threshold[node_index]
        index_X_child_l = X_target_node[:,phi] <= threshold
        X_target_child_l = X_target_node[index_X_child_l,:]
        Y_target_child_l = Y_target_node[index_X_child_l]
        STRUT(decisiontree,
              tree.children_left[node_index],
              X_target_child_l,
              Y_target_child_l)        
                
    if tree.children_right[node_index] != -1:
        threshold = tree.threshold[node_index]
        index_X_child_r = X_target_node[:,phi] > threshold
        X_target_child_r = X_target_node[index_X_child_r,:]
        Y_target_child_r = Y_target_node[index_X_child_r]
        STRUT(decisiontree,
              tree.children_right[node_index],
              X_target_child_r,
              Y_target_child_r)

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
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)

    # Plot the classification threshold
    plt.plot(X[0:dataset_length//2,0],X[0:dataset_length//2,1],"ro")
    plt.plot(X[dataset_length//2:,0],X[dataset_length//2:,1],"bo")
    for node,feature in enumerate(clf.tree_.feature):
        if feature ==0:
            plt.axvline(x = clf.tree_.threshold[node])
        elif feature == 1:
            plt.axhline(y = clf.tree_.threshold[node])
    plt.show()
    # plot the tree
    dot_data = tree.export_graphviz(clf, out_file=None, 
                             feature_names=["feature_"+str(i) for i in range(D)],  
                             class_names=["class_0", "class_1"],  
                             filled=True, rounded=True,  
                             special_characters=True)  

    graph = graphviz.Source(dot_data)  
    graph.render('prarent_tree.gv', view=True) 
    # Build a new dataset
    X = np.random.randn(dataset_length,D+1)*0.1
    #X[:,1] = np.nan
    X[0:dataset_length//2,0] += 0.3
    Y = np.ones(dataset_length)
    Y[0:dataset_length//2] *= 0
    # Call STRUT
    
    STRUT(clf,0,X,Y)
    # Plot the new thresholds
    
    plt.plot(X[0:dataset_length//2,0],X[0:dataset_length//2,1],"ro")
    plt.plot(X[dataset_length//2:,0],X[dataset_length//2:,1],"bo")
    for node,feature in enumerate(clf.tree_.feature):
        if feature ==0:
            plt.axvline(x = clf.tree_.threshold[node])
        elif feature == 1:
            plt.axhline(y = clf.tree_.threshold[node])
    plt.show()
    # Plot the new tree
    dot_data = tree.export_graphviz(clf, out_file=None, 
                             feature_names=["feature_"+str(i) for i in range(D)],  
                             class_names=["class_0", "class_1"],  
                             filled=True, rounded=True,  
                             special_characters=True)  

    graph = graphviz.Source(dot_data)  
    graph.render('child_tree.gv', view=True)   
