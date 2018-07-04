#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 01:50:08 2018

@author: mounir
"""

import numpy as np
import sklearn

from Generator import ClusterPoints,StreamGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from Changes import apply_drift, delete_clusters, change_cluster_weight, create_new_clusters, apply_density_change
import pandas as pd

import sys
sys.path.insert(0, "..")    
#sys.path.insert(0, "../data_mngmt/")
#sys.path.insert(0, "../utils/")
import SER_rec
import STRUT
import sklearn.ensemble as skl_ens
import pickle



# =============================================================================
# 
# =============================================================================
def del_cl_prop(sg,N,cl,p0):
    for i in range(N):
        ind = np.where(np.array(sg.class_labels) == cl)[0]
        delete_clusters(sg.clusters, [ind[0]])
        sg.weights = np.delete(sg.weights,ind[0])
        sg.class_labels = np.delete(sg.class_labels,ind[0]).tolist()
        
    N0 = sg.class_labels.count(0)
    N1 = sg.class_labels.count(1)
    
    for i,clust in enumerate(sg.clusters) :
        if clust.class_label == 0:
            clust.weight = p0/N0
            sg.weights[i] = p0/N0
        else:
            clust.weight = (1-p0)/N1
            sg.weights[i] = (1-p0)/N1
            
def new_cl_prop(sg,N,cl,p0):

    args_ = {'weight' : 1, 'dimensionality' : sg.dimensionality, 'min_projected_dim' : sg.dimensionality, 
            'max_projected_dim' : sg.dimensionality, 'min_coordinate' : sg.min_coordinate, 'max_coordinate' : sg.max_coordinate, 
            'min_projected_dim_var' : sg.max_projected_dim_var, 'max_projected_dim_var': sg.max_projected_dim_var, 'class_label' : cl}
   
    for k in range(N):
        sg.weights = np.concatenate((sg.weights,[ args_['weight']]))
        sg.class_labels.append( args_['class_label'] )
        create_new_clusters(sg.clusters, [args_])
        
    N0 = sg.class_labels.count(0)
    N1 = sg.class_labels.count(1)
    
    for i,clust in enumerate(sg.clusters) :
        if clust.class_label == 0:
            clust.weight = p0/N0
            sg.weights[i] = p0/N0
        else:
            clust.weight = (1-p0)/N1
            sg.weights[i] = (1-p0)/N1
            
def random_drift_cl(sg,var,nb):
    
    size = var
    cfs = size*(2*np.random.rand(len(sg.clusters)) -1)
    cfs_phi = np.random.choice(np.linspace(0,sg.dimensionality-1,sg.dimensionality).astype(int),len(sg.clusters))
    
    
    cluster_feature_speed = {i: {cfs_phi[i]: cfs[i]} for i in range(nb)}
    apply_drift(sg.clusters, cluster_feature_speed,sg.min_coordinate, sg.max_coordinate)
    
def random_var_cl(sg,var,nb):
    size = var
    cfs = size*(2*np.random.rand(len(sg.clusters)) -1)
    cfs_phi = np.random.choice(np.linspace(0,sg.dimensionality-1,sg.dimensionality).astype(int),len(sg.clusters))
    
    
    cluster_feature_std = {i: {cfs_phi[i]: cfs[i]} for i in range(nb)}
    apply_density_change(sg.clusters, cluster_feature_std)
    
# =============================================================================
#             
# =============================================================================

N_EXP = 19
N_REP = 10

path_out = './outputs/final/'


for i in range(N_REP):
    N1 = 8
    N0 = 8
    
    prop1 = 0.5
    prop1 = prop1/N1
    prop0 = 0.5
    prop0 = prop0/N0
    
    dim = 3
    space_bound = 10
    
    n_source = 500
    
    var = "-"
    sg = StreamGenerator(number_points=n_source,
    					 weights=[prop1,prop1,prop1,prop1,prop1,prop1,prop1,prop1,prop0,prop0, prop0,prop0,prop0,prop0, prop0,prop0],
    					 dimensionality=dim,
    					 min_projected_dim=dim,
    					 max_projected_dim=dim,
    					 min_coordinate=-space_bound,
    					 max_coordinate=space_bound,
    					 min_projected_dim_var=2,
    					 max_projected_dim_var=5,
    					 class_labels=[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    					 )
    pickle.dump(sg,open(path_out+"source_generator_"+sg.get_file_name()+"Exp"+str(N_EXP)+"std"+str(var)+"_Rep"+str(i)+".pkl","wb"))
    
    #Mise en forme du source
    
    a = sg.get_full_dataset(n_source)
    X_source = a[1].values[:,:-1]
    Y_source = a[1].values[:,-1]
    
    a[1].to_csv(path_out+"DataTrain_source_"+sg.get_file_name()+"Exp"+str(N_EXP)+"std"+str(var)+"_Rep"+str(i)+".csv")

#    if unit_test:
#        sns.pairplot(a[1],hue="cluster")
#        plt.savefig(path_out+'source.pdf')
#        plt.show()
    
    
    print('N clust init:',len(sg.clusters))
    
    
    
    #Mise en forme du target :
    
        
    N1 = 8
    N0 = 8
    
    prop1 = 0.05
    prop1 = prop1/N1
    prop0 = 0.95
    prop0 = prop0/N0



    #New clusters
    
    new_cl_prop(sg,1,0,0.95) 
    new_cl_prop(sg,1,1,0.95) 
    del_cl_prop(sg,1,0,0.95)
    del_cl_prop(sg,1,1,0.95)


    random_drift_cl(sg,1,N1+N0)
    #random_var_cl(sg,5,N1+N0)

    
    
    #==============================================================================
    # 
    #==============================================================================

    print('N clust target:',len(sg.clusters))
    
    pickle.dump(sg,open(path_out+"target_generator_"+sg.get_file_name()+"Exp"+str(N_EXP)+"std"+str(var)+"_Rep"+str(i)+".pkl","wb"))
    
    c = sg.get_full_dataset(1000)
    X_target_test = c[1].values[:,:-1]
    Y_target_test = c[1].values[:,-1]
    
    c[1].to_csv(path_out+"DataTest_target_"+sg.get_file_name()+"Exp"+str(N_EXP)+"std"+str(var)+"_Rep"+str(i)+".csv")

    
#    if unit_test:
#        sns.pairplot( c[1],hue="cluster")
#        plt.savefig(path_out+'test.pdf')
#        plt.show()  
     
    b = sg.get_full_dataset(200)
    
    b[1].to_csv(path_out+"DataTrain_target_"+sg.get_file_name()+"Exp"+str(N_EXP)+"std"+str(var)+"_Rep"+str(i)+".csv")

    X_target_train = b[1].values[:,:-1]
    Y_target_train = b[1].values[:,-1]
    
#    if unit_test:
#        sns.pairplot( b[1],hue="cluster")
#        plt.savefig(path_out+'train.pdf')
#        plt.show()
