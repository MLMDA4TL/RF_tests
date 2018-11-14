#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 01:50:08 2018

@author: mounir
"""

import sklearn
import numpy as np

import sklearn.ensemble as skl_ens

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


from collections import namedtuple

score = namedtuple("Score", "algo max_depth nb_tree data auc score_rate tpr fpr f1 n_iter params comments")

def write_score(scores_file, score):
# scores_df = pd.read_csv(scores_file, sep=',', header=0)
    new_score_df = pd.DataFrame([[score.algo, score.max_depth, score.nb_tree,
    score.data, score.auc, score.score_rate, score.tpr,
    score.fpr, score.f1, score.n_iter, score.params, score.comments]])
    with open(scores_file, 'a') as f:
        new_score_df.to_csv(f, header=False, index=False)
        f.close()


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
      
def change_prop(sg,weights):
    labels = list(set(sg.class_labels))
    N = len(labels)
    
    nb_cl = np.zeros(N)
    
    for i in range(N):
        nb_cl[i] = sg.class_labels.count(labels[i])
        sg.weights[sg.class_labels == labels[i]] = weights[i]/nb_cl[i]

        clusts = list(np.array(sg.clusters)[sg.class_labels == labels[i]])
        for c in clusts:
            c.weight = weights[i]/nb_cl[i]
        
    #sg.compute_probabilities_draw_cluster()

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
            
def random_drift_cl(sg,size,nb):
    
    cfs = size*(2*np.random.rand(len(sg.clusters)) -1)
    cfs_phi = np.random.choice(np.linspace(0,sg.dimensionality-1,sg.dimensionality).astype(int),len(sg.clusters))
    
    
    cluster_feature_speed = {i: {cfs_phi[i]: cfs[i]} for i in range(nb)}
    apply_drift(sg.clusters, cluster_feature_speed,sg.min_coordinate, sg.max_coordinate)
    
def random_var_cl(sg,factor_min,factor_max,nb,clust = None):
    
    cfs = (factor_max - factor_min)*np.random.rand(len(sg.clusters)) + factor_min
    
    if clust is None :
        cfs_phi = np.random.choice(np.linspace(0,sg.dimensionality-1,sg.dimensionality).astype(int),len(sg.clusters))
    else:
        cfs_phi = clust
    
    
    cluster_feature_std = {i: {cfs_phi[i]: cfs[i]} for i in range(nb)}
    
    apply_density_change(sg.clusters, cluster_feature_std)

def change_cl_prop(sg,factor_min,factor_max,nb):
    
    cfs = (factor_max - factor_min)*np.random.rand(len(sg.clusters)) + factor_min
    cfs_phi = np.random.choice(np.linspace(0,sg.dimensionality-1,sg.dimensionality).astype(int),len(sg.clusters))
    
    
    cluster_feature_std = {i: {cfs_phi[i]: cfs[i]} for i in range(nb)}
    
    apply_density_change(sg.clusters, cluster_feature_std)    
# =============================================================================
#             
# =============================================================================
unit_test = 1 
plots = 0
WRITE_SCORE = 0

N_EXP = 11
N_REP = 10

#if unit_test:
#    print('Test unique')
#    N_REP = 1

path_out = './outputs/final/'

score_rf = np.zeros(N_REP)
score_rf_target = np.zeros(N_REP)
score_rf_strut = np.zeros(N_REP)
####
score_rf_strut_adapt = np.zeros(N_REP)
score_rf_strut_noprune_update = np.zeros(N_REP)
score_rf_strut_noprune_noupdate = np.zeros(N_REP)
####
score_rf_ser = np.zeros(N_REP)
score_rf_ser_no_red = np.zeros(N_REP)
###
score_rf_ser_no_ext = np.zeros(N_REP)
score_rf_ser_ext_cond = np.zeros(N_REP)
score_rf_ser_no_red_no_ext = np.zeros(N_REP)
score_rf_ser_no_red_no_ext_cond = np.zeros(N_REP)
####

fpr_rf = np.zeros(N_REP)
fpr_rf_target = np.zeros(N_REP)
fpr_rf_strut = np.zeros(N_REP)
####
fpr_rf_strut_adapt = np.zeros(N_REP)
fpr_rf_strut_noprune_update = np.zeros(N_REP)
fpr_rf_strut_noprune_noupdate = np.zeros(N_REP)
####
fpr_rf_ser = np.zeros(N_REP)
fpr_rf_ser_no_red = np.zeros(N_REP)
####
fpr_rf_ser_no_ext = np.zeros(N_REP)
fpr_rf_ser_ext_cond = np.zeros(N_REP)
fpr_rf_ser_no_red_no_ext = np.zeros(N_REP)
fpr_rf_ser_no_red_no_ext_cond = np.zeros(N_REP)

tpr_rf = np.zeros(N_REP)
tpr_rf_target = np.zeros(N_REP)
tpr_rf_strut = np.zeros(N_REP)
####
tpr_rf_strut_adapt = np.zeros(N_REP)
tpr_rf_strut_noprune_update = np.zeros(N_REP)
tpr_rf_strut_noprune_noupdate = np.zeros(N_REP)
####
tpr_rf_ser = np.zeros(N_REP)
tpr_rf_ser_no_red = np.zeros(N_REP)
####
tpr_rf_ser_no_ext = np.zeros(N_REP)
tpr_rf_ser_ext_cond = np.zeros(N_REP)
tpr_rf_ser_no_red_no_ext = np.zeros(N_REP)
tpr_rf_ser_no_red_no_ext_cond = np.zeros(N_REP)
        
auc_rf = np.zeros(N_REP)
auc_rf_target = np.zeros(N_REP)
auc_rf_strut = np.zeros(N_REP)
####
auc_rf_strut_adapt = np.zeros(N_REP)
auc_rf_strut_noprune_update = np.zeros(N_REP)
auc_rf_strut_noprune_noupdate = np.zeros(N_REP)
####
auc_rf_ser = np.zeros(N_REP)
auc_rf_ser_no_red = np.zeros(N_REP)
####
auc_rf_ser_no_ext = np.zeros(N_REP)
auc_rf_ser_ext_cond = np.zeros(N_REP)
auc_rf_ser_no_red_no_ext = np.zeros(N_REP)
auc_rf_ser_no_red_no_ext_cond = np.zeros(N_REP)
    
f1_rf = np.zeros(N_REP)
f1_rf_target = np.zeros(N_REP)
f1_rf_strut = np.zeros(N_REP)
####
f1_rf_strut_adapt = np.zeros(N_REP)
f1_rf_strut_noprune_update = np.zeros(N_REP)
f1_rf_strut_noprune_noupdate = np.zeros(N_REP)
####
f1_rf_ser = np.zeros(N_REP)
f1_rf_ser_no_red = np.zeros(N_REP)
###
f1_rf_ser_no_ext = np.zeros(N_REP)
f1_rf_ser_ext_cond = np.zeros(N_REP)
f1_rf_ser_no_red_no_ext = np.zeros(N_REP)
f1_rf_ser_no_red_no_ext_cond = np.zeros(N_REP)
###


for i in range(N_REP):
    N1 = 15
    N0 = 15
    
    prop1_init = 0.5
    prop0_init = 0.5

    
    dim = 3
    space_bound = 40
    
    n_source = 200
    minV = 5
    maxV = 15
    var = 3
    name_param = "std"

    prop1_init = prop1_init/N1
    prop0_init = prop0_init/N0
    
    weights = list(np.concatenate((np.repeat(prop1_init,N1),np.repeat(prop0_init,N0))))
    class_labels = list(np.concatenate((np.repeat(1,N1),np.repeat(0,N0))))    

       
    sg = StreamGenerator(number_points=n_source,
    					 weights=weights,
    					 dimensionality=dim,
    					 min_projected_dim=dim,
    					 max_projected_dim=dim,
    					 min_coordinate=-space_bound,
    					 max_coordinate=space_bound,
    					 min_projected_dim_var=minV,
    					 max_projected_dim_var=maxV,
    					 class_labels= class_labels,
    					 )

    params_init = 'D'+str(dim)+'space_size'+str(space_bound)+'nclustinit' +str(len(sg.clusters))+'prop_init'+ \
       str(prop0_init*N0)+'var'+str(minV)+'-'+str(maxV)+'n_source'+str(n_source)
       
    if not unit_test:
        pickle.dump(sg,open(path_out+"source_generator_"+sg.get_file_name()+"Exp"+str(N_EXP)+name_param+str(var)+"_Rep"+str(i)+".pkl","wb"))
    
    #Mise en forme du source
    
    a = sg.get_full_dataset(n_source)
    X_source = a[1].values[:,:-1]
    Y_source = a[1].values[:,-1]
    
    if not unit_test:
        a[1].to_csv(path_out+"DataTrain_source_"+sg.get_file_name()+"Exp"+str(N_EXP)+name_param+str(var)+"_Rep"+str(i)+".csv")

    if plots:
        sns.pairplot(a[1],hue="cluster")
        plt.savefig(path_out+'source.pdf')
        plt.show()
    
    
    print('N clust init:',len(sg.clusters))
    
    # =============================================================================
    #     Mise en forme du target :
    # =============================================================================
    var_change = False
    ampl_var_change_m = 0.5 
    ampl_var_change_M = 2 
    drift = True
    v_drift = minV
    n_clust_change = False
    
    #Only prop changes:
    
    prop1 = 0.05
    prop0 = 0.95
    transform = "prop_only"+str(prop0)
    
    
    change_prop(sg,[prop0,prop1])

    # Source Clusters Changes
    N0_T = 5
    N1_T = 5
    N0_S = 0
    N1_S = 0
    
    if n_clust_change :
        transform = 'n_clust('+str(N0)+','+str(N1)+')->('+str(N0+N0_T-N0_S)+','+str(N1+N1_T-N1_S)+')'
        if ( N0_S > 0 or N1_S > 0 ) :
            del_cl_prop(sg,N0_S,0,prop0)
            del_cl_prop(sg,N1_S,1,prop0)
        if ( N0_T > 0 or N1_T > 0 ) :
            new_cl_prop(sg,N0_T,0,prop0) 
            new_cl_prop(sg,N1_T,1,prop0)

#    new_cl_prop(sg,1,0,0.95) 
#    new_cl_prop(sg,1,1,0.95) 
#    del_cl_prop(sg,1,0,0.95)
#    del_cl_prop(sg,1,1,0.95)

    #Clusters Weights Changes : 

    # Drift and Stretching : 
    if drift:
        transform = "drift"+str(v_drift)
        random_drift_cl(sg,v_drift,len(sg.class_labels))
    if var_change :
        transform = "var_change"+str(ampl_var_change_m)+'-'+str(ampl_var_change_M)
        random_var_cl(sg,ampl_var_change_m,ampl_var_change_M,len(sg.class_labels))

    
    
    #==============================================================================
    # 
    #==============================================================================
    params_fin = 'nclust' +str(len(sg.clusters))+'prop'+ \
       str(prop0)+'var'+str(minV)+'-'+str(maxV)+'n_source'+str(n_source)+'TRANSFORM--'+transform
       
    print('N clust target:',len(sg.clusters))
    
    if not unit_test:
        pickle.dump(sg,open(path_out+"target_generator_"+sg.get_file_name()+"Exp"+str(N_EXP)+name_param+str(var)+"_Rep"+str(i)+".pkl","wb"))
    
    
    c = sg.get_full_dataset(10000)
    X_target_test = c[1].values[:,:-1]
    Y_target_test = c[1].values[:,-1]

    if not unit_test:
        c[1].to_csv(path_out+"DataTest_target_"+sg.get_file_name()+"Exp"+str(N_EXP)+name_param+str(var)+"_Rep"+str(i)+".csv")

    
    if plots:
        sns.pairplot( c[1],hue="cluster")
        plt.savefig(path_out+'test.pdf')
        plt.show()  
     
    b = sg.get_full_dataset(n_source)
    
    if not unit_test:
        b[1].to_csv(path_out+"DataTrain_target_"+sg.get_file_name()+"Exp"+str(N_EXP)+name_param+str(var)+"_Rep"+str(i)+".csv")

    X_target_train = b[1].values[:,:-1]
    Y_target_train = b[1].values[:,-1]
    
    if plots:
        sns.pairplot( b[1],hue="cluster")
        plt.savefig(path_out+'train.pdf')
        plt.show()
        
# =============================================================================
#   Tests SER/STRUT basiques pour l'intuition :
# =============================================================================
    MAX_DEPTH = None
    bootstrap = True
    N_EST = 10
    

    
    if unit_test:
        rf = skl_ens.RandomForestClassifier(n_estimators = N_EST,max_depth = MAX_DEPTH, bootstrap = bootstrap  )
        
        rf_th = skl_ens.RandomForestClassifier(n_estimators = N_EST, bootstrap = bootstrap  )
        rf_th.fit(X_target_test,Y_target_test)
        
        rf_target = skl_ens.RandomForestClassifier(n_estimators = N_EST, bootstrap = bootstrap )
        
        rf.fit(X_source,Y_source)
        rf_target.fit(X_target_train,Y_target_train)
        # =============================================================================
        #               SER
        # =============================================================================
        rf_ser = SER_rec.SER_RF(rf, X_target_train, Y_target_train)
        rf_ser_no_red = SER_rec.SER_RF(rf, X_target_train, Y_target_train, no_red_on_cl = True, cl_no_red  = [1])
        
        ###
        rf_ser_no_ext = SER_rec.SER_RF(rf, X_target_train, Y_target_train, no_red_on_cl = False, no_ser_on_cl= True, cl_no_ser= [1])
        rf_ser_ext_cond = SER_rec.SER_RF(rf, X_target_train, Y_target_train, no_red_on_cl = False, no_ser_on_cl= True, cl_no_ser= [1], exp_refinement = True)
        rf_ser_no_red_no_ext = SER_rec.SER_RF(rf, X_target_train, Y_target_train, no_red_on_cl = True, cl_no_red  = [1],no_ser_on_cl= True, cl_no_ser= [1])
        rf_ser_no_red_no_ext_cond = SER_rec.SER_RF(rf, X_target_train, Y_target_train, no_red_on_cl = True, cl_no_red  = [1],no_ser_on_cl= True, cl_no_ser= [1], exp_refinement = True)
        ###
        
        # =============================================================================
        #               STRUT
        # =============================================================================
        
        #rf_strut = STRUT.STRUT_RF(rf, X_target_train, Y_target_train,1)
        rf_strut = STRUT.STRUT_RF(rf,X_target_train,Y_target_train, pruning_updated_node=True, no_prune_on_cl=False, cl_no_prune=None, prune_lone_instance=True)
        rf_strut_adapt = STRUT.STRUT_RF(rf,X_target_train,Y_target_train, pruning_updated_node=True, no_prune_on_cl=False, cl_no_prune=None, prune_lone_instance=True, adapt_prop=True)

        #rf_strut_noprune_update = STRUT.STRUT_RF(rf,X_target_train,Y_target_train, pruning_updated_node=True, no_prune_on_cl=True, cl_no_prune=[1], prune_lone_instance=False)
        #rf_strut_noprune_noupdate = STRUT.STRUT_RF(rf,X_target_train,Y_target_train, pruning_updated_node=False, no_prune_on_cl=True, cl_no_prune=[1], prune_lone_instance=False)
        #rf_strut = STRUT.STRUT_RF(rf,X_target_train,Y_target_train, pruning_updated_node=True, no_prune_on_cl=False, cl_no_prune=None, prune_lone_instance=True)

        # =============================================================================
        #         
        # =============================================================================
 
# =============================================================================
# 
# =============================================================================        
        p0_rf = rf.estimators_[0].tree_.value[rf.estimators_[0].tree_.feature == -2][:,:,0].reshape(-1)
        p0_rf = p0_rf/sum(p0_rf)
        p1_rf = rf.estimators_[0].tree_.value[rf.estimators_[0].tree_.feature == -2][:,:,1].reshape(-1)
        p1_rf = p1_rf/sum(p1_rf)
        

        
        leaves_rf = list(np.argmax(rf.estimators_[0].tree_.value[rf.estimators_[0].tree_.feature == -2],axis=2).reshape(-1))
        leaves_rf_target = list(np.argmax(rf_target.estimators_[0].tree_.value[rf_target.estimators_[0].tree_.feature == -2],axis=2).reshape(-1))

        leaves_rf_ser = list(np.argmax(rf_ser.estimators_[0].tree_.value[rf_ser.estimators_[0].tree_.feature == -2],axis=2).reshape(-1))
        leaves_rf_ser_no_red = list(np.argmax(rf_ser_no_red.estimators_[0].tree_.value[rf_ser_no_red.estimators_[0].tree_.feature == -2],axis=2).reshape(-1))
        #leaves_rf = list(np.argmax(rf.estimators_[0].tree_.value[rf.estimators_[0].tree_.feature == -2],axis=2).reshape(-1)).count(1)

        alpha = 0.1
        beta = 0.5*alpha
        n = 200
        
        x = np.linspace(0,leaves_rf.count(1),leaves_rf.count(1))
        qi = np.power(1 - alpha*p1_rf,n*beta)
        qi_or = np.power(1 - p1_rf,n*(beta/alpha))
        
        ###plt.plot(x,qi[np.array(leaves_rf) == 1], x,qi_or[np.array(leaves_rf) == 1])
        
        
        #from sklearn.tree import export_graphviz
        #import graphviz
        #dot_data = export_graphviz(rf.estimators_[0], "dtree.dot")
# =============================================================================
#         
# =============================================================================
        
        score_rf[i] = rf.score(X_target_test,Y_target_test)
        score_rf_target[i] = rf_target.score(X_target_test,Y_target_test)
        score_rf_strut[i] = rf_strut.score(X_target_test,Y_target_test)
        ####
        score_rf_strut_adapt[i] = rf_strut_adapt.score(X_target_test,Y_target_test)
        #score_rf_strut_noprune_update[i] = rf_strut_noprune_update.score(X_target_test,Y_target_test)
        #score_rf_strut_noprune_noupdate[i] = rf_strut_noprune_noupdate.score(X_target_test,Y_target_test)
        ####
        score_rf_ser[i] = rf_ser.score(X_target_test,Y_target_test)
        score_rf_ser_no_red[i] = rf_ser_no_red.score(X_target_test,Y_target_test)
        ###
        score_rf_ser_no_ext[i] = rf_ser_no_ext.score(X_target_test,Y_target_test)
        score_rf_ser_ext_cond[i] = rf_ser_ext_cond.score(X_target_test,Y_target_test)
        score_rf_ser_no_red_no_ext[i] = rf_ser_no_red_no_ext.score(X_target_test,Y_target_test)
        score_rf_ser_no_red_no_ext_cond[i] = rf_ser_no_red_no_ext_cond.score(X_target_test,Y_target_test)
        ####

        a, b, t = sklearn.metrics.roc_curve(Y_target_test, rf.predict_proba(X_target_test)[:,1]) 
        index = np.argmin(np.abs(t - 0.5))
        fpr_rf[i], tpr_rf[i] = a[index], b[index]

        a,b,t = sklearn.metrics.roc_curve(Y_target_test, rf_target.predict_proba(X_target_test)[:,1])
        index = np.argmin(np.abs(t - 0.5))
        fpr_rf_target[i], tpr_rf_target[i] = a[index], b[index]
        
        a,b,t = sklearn.metrics.roc_curve(Y_target_test, rf_strut.predict_proba(X_target_test)[:,1])
        index = np.argmin(np.abs(t - 0.5))
        fpr_rf_strut[i], tpr_rf_strut[i] = a[index], b[index] 

        a,b,t = sklearn.metrics.roc_curve(Y_target_test, rf_strut_adapt.predict_proba(X_target_test)[:,1])
        index = np.argmin(np.abs(t - 0.5))
        fpr_rf_strut_adapt[i], tpr_rf_strut_adapt[i] = a[index], b[index]
        #a,b,t = sklearn.metrics.roc_curve(Y_target_test, rf_strut_noprune_update.predict_proba(X_target_test)[:,1])
        #index = np.argmin(np.abs(t - 0.5))
        #fpr_rf_strut_noprune_update[i], tpr_rf_strut_noprune_update[i] = a[index], b[index]

        #a,b,t = sklearn.metrics.roc_curve(Y_target_test, rf_strut_noprune_noupdate.predict_proba(X_target_test)[:,1])
        #index = np.argmin(np.abs(t - 0.5))
        #fpr_rf_strut_noprune_noupdate[i], tpr_rf_strut_noprune_noupdate[i] = a[index], b[index]
        
        a,b,t = sklearn.metrics.roc_curve(Y_target_test, rf_ser.predict_proba(X_target_test)[:,1])
        index = np.argmin(np.abs(t - 0.5))
        fpr_rf_ser[i], tpr_rf_ser[i] = a[index], b[index] 
        
        a,b,t = sklearn.metrics.roc_curve(Y_target_test, rf_ser_no_red.predict_proba(X_target_test)[:,1])
        index = np.argmin(np.abs(t - 0.5))
        fpr_rf_ser_no_red[i], tpr_rf_ser_no_red[i] = a[index], b[index]  
        
        ####
        a,b,t = sklearn.metrics.roc_curve(Y_target_test, rf_ser_no_ext.predict_proba(X_target_test)[:,1])
        index = np.argmin(np.abs(t - 0.5))
        fpr_rf_ser_no_ext[i], tpr_rf_ser_no_ext[i] = a[index], b[index]  
        
        a,b,t = sklearn.metrics.roc_curve(Y_target_test, rf_ser_ext_cond.predict_proba(X_target_test)[:,1])
        index = np.argmin(np.abs(t - 0.5))
        fpr_rf_ser_ext_cond[i], tpr_rf_ser_ext_cond[i] = a[index], b[index] 
        
        a,b,t = sklearn.metrics.roc_curve(Y_target_test, rf_ser_no_red_no_ext.predict_proba(X_target_test)[:,1])
        index = np.argmin(np.abs(t - 0.5))
        fpr_rf_ser_no_red_no_ext[i], tpr_rf_ser_no_red_no_ext[i] = a[index], b[index] 
        
        a,b,t = sklearn.metrics.roc_curve(Y_target_test, rf_ser_no_red_no_ext_cond.predict_proba(X_target_test)[:,1])
        index = np.argmin(np.abs(t - 0.5))
        fpr_rf_ser_no_red_no_ext_cond[i], tpr_rf_ser_no_red_no_ext_cond[i] = a[index], b[index] 
        ####    
        
        auc_rf[i] = sklearn.metrics.roc_auc_score(Y_target_test, rf.predict_proba(X_target_test)[:,1])
        auc_rf_target[i] = sklearn.metrics.roc_auc_score(Y_target_test, rf_target.predict_proba(X_target_test)[:,1])
        auc_rf_strut[i] = sklearn.metrics.roc_auc_score(Y_target_test, rf_strut.predict_proba(X_target_test)[:,1])
        ####
        auc_rf_strut_adapt[i] = sklearn.metrics.roc_auc_score(Y_target_test, rf_strut_adapt.predict_proba(X_target_test)[:,1])
        #auc_rf_strut_noprune_update[i] = sklearn.metrics.roc_auc_score(Y_target_test, rf_strut_noprune_update.predict_proba(X_target_test)[:,1])
        #auc_rf_strut_noprune_noupdate[i] = sklearn.metrics.roc_auc_score(Y_target_test, rf_strut_noprune_noupdate.predict_proba(X_target_test)[:,1])
        ####
        auc_rf_ser[i] = sklearn.metrics.roc_auc_score(Y_target_test, rf_ser.predict_proba(X_target_test)[:,1])
        auc_rf_ser_no_red[i] = sklearn.metrics.roc_auc_score(Y_target_test, rf_ser_no_red.predict_proba(X_target_test)[:,1])
        
        ####
        auc_rf_ser_no_ext[i] = sklearn.metrics.roc_auc_score(Y_target_test, rf_ser_no_ext.predict_proba(X_target_test)[:,1])
        auc_rf_ser_ext_cond[i] = sklearn.metrics.roc_auc_score(Y_target_test, rf_ser_ext_cond.predict_proba(X_target_test)[:,1])
        auc_rf_ser_no_red_no_ext[i] = sklearn.metrics.roc_auc_score(Y_target_test, rf_ser_no_red_no_ext.predict_proba(X_target_test)[:,1])
        auc_rf_ser_no_red_no_ext_cond[i] = sklearn.metrics.roc_auc_score(Y_target_test, rf_ser_no_red_no_ext_cond.predict_proba(X_target_test)[:,1])
        ####
    
        f1_rf[i] = sklearn.metrics.f1_score(Y_target_test, rf.predict(X_target_test))
        f1_rf_target[i] = sklearn.metrics.f1_score(Y_target_test, rf_target.predict(X_target_test))
        f1_rf_strut[i] = sklearn.metrics.f1_score(Y_target_test, rf_strut.predict(X_target_test))
        ####
        f1_rf_strut_adapt[i] = sklearn.metrics.f1_score(Y_target_test, rf_strut_adapt.predict(X_target_test))
        #f1_rf_strut_noprune_update[i] = sklearn.metrics.f1_score(Y_target_test, rf_strut_noprune_update.predict(X_target_test))
        #f1_rf_strut_noprune_noupdate[i] = sklearn.metrics.f1_score(Y_target_test, rf_strut_noprune_noupdate.predict(X_target_test))
        ####
        f1_rf_ser[i] = sklearn.metrics.f1_score(Y_target_test, rf_ser.predict(X_target_test))
        f1_rf_ser_no_red[i] = sklearn.metrics.f1_score(Y_target_test, rf_ser_no_red.predict(X_target_test))
        
        ###
        f1_rf_ser_no_ext[i] = sklearn.metrics.f1_score(Y_target_test, rf_ser_no_ext.predict(X_target_test))
        f1_rf_ser_ext_cond[i] = sklearn.metrics.f1_score(Y_target_test, rf_ser_ext_cond.predict(X_target_test))
        f1_rf_ser_no_red_no_ext[i] = sklearn.metrics.f1_score(Y_target_test, rf_ser_no_red_no_ext.predict(X_target_test))
        f1_rf_ser_no_red_no_ext_cond[i] = sklearn.metrics.f1_score(Y_target_test, rf_ser_no_red_no_ext_cond.predict(X_target_test))
        ###
        

scores_file = './scores_file.csv'
if MAX_DEPTH == None : 
    score.max_depth = "None" 
else:
    score.max_depth = MAX_DEPTH  
 
score.nb_tree = N_EST    
score.n_iter = N_REP   
score.data = 'SYNTH_IMB'+transform+'_prop'+str(prop1)
score.comments = ""

score.params = params_init+'_INTO_'+params_fin

if WRITE_SCORE:
    # =============================================================================
    #  
    # =============================================================================
        
    score.algo = "source"
    score.score_rate = np.mean(score_rf)
    score.tpr = np.mean(tpr_rf)
    score.fpr = np.mean(fpr_rf)
    score.f1 = np.mean(f1_rf)
    score.auc = np.mean(auc_rf)
    write_score(scores_file, score)
    
    score.algo = "target"
    score.score_rate = np.mean(score_rf_target)
    score.tpr = np.mean(tpr_rf_target)
    score.fpr = np.mean(fpr_rf_target)
    score.f1 = np.mean(f1_rf_target)
    score.auc = np.mean(auc_rf_target)
    write_score(scores_file, score)
    
    # =============================================================================
    #                           SER
    # =============================================================================
    
    score.algo = "ser"
    score.score_rate = np.mean(score_rf_ser)
    score.tpr = np.mean(tpr_rf_ser)
    score.fpr = np.mean(fpr_rf_ser)
    score.f1 = np.mean(f1_rf_ser)
    score.auc = np.mean(auc_rf_ser)
    write_score(scores_file, score)
    
    score.algo = "ser no red"
    score.score_rate = np.mean(score_rf_ser_no_red)
    score.tpr = np.mean(tpr_rf_ser_no_red)
    score.fpr = np.mean(fpr_rf_ser_no_red)
    score.f1 = np.mean(f1_rf_ser_no_red)
    score.auc = np.mean(auc_rf_ser_no_red)
    write_score(scores_file, score)
    
    score.algo = "ser no ext"
    score.score_rate = np.mean(score_rf_ser_no_ext)
    score.tpr = np.mean(tpr_rf_ser_no_ext)
    score.fpr = np.mean(fpr_rf_ser_no_ext)
    score.f1 = np.mean(f1_rf_ser_no_ext)
    score.auc = np.mean(auc_rf_ser_no_ext)
    write_score(scores_file, score)
    
    score.algo = "ser no ext cond"
    score.score_rate = np.mean(score_rf_ser_ext_cond)
    score.tpr = np.mean(tpr_rf_ser_ext_cond)
    score.fpr = np.mean(fpr_rf_ser_ext_cond)
    score.f1 = np.mean(f1_rf_ser_ext_cond)
    score.auc = np.mean(auc_rf_ser_ext_cond)
    write_score(scores_file, score)
    
    score.algo = "ser no red no ext"
    score.score_rate = np.mean(score_rf_ser_no_red_no_ext)
    score.tpr = np.mean(tpr_rf_ser_no_red_no_ext)
    score.fpr = np.mean(fpr_rf_ser_no_red_no_ext)
    score.f1 = np.mean(f1_rf_ser_no_red_no_ext)
    score.auc = np.mean(auc_rf_ser_no_red_no_ext)
    write_score(scores_file, score)
    
    score.algo = "ser no red no ext cond"
    score.score_rate = np.mean(score_rf_ser_no_red_no_ext_cond)
    score.tpr = np.mean(tpr_rf_ser_no_red_no_ext_cond)
    score.fpr = np.mean(fpr_rf_ser_no_red_no_ext_cond)
    score.f1 = np.mean(f1_rf_ser_no_red_no_ext_cond)
    score.auc = np.mean(auc_rf_ser_no_red_no_ext_cond)
    write_score(scores_file, score)
    
    # =============================================================================
    #                       STRUT
    # =============================================================================
    
    
    score.algo = "strut"
    score.score_rate = np.mean(score_rf_strut)
    score.tpr = np.mean(tpr_rf_strut)
    score.fpr = np.mean(fpr_rf_strut)
    score.f1 = np.mean(f1_rf_strut)
    score.auc = np.mean(auc_rf_strut)
    write_score(scores_file, score)
    
    #score.algo = "strut noprune update"
    #score.score_rate = np.mean(score_rf_strut_noprune_update)
    #score.tpr = np.mean(tpr_rf_strut_noprune_update)
    #score.fpr = np.mean(fpr_rf_strut_noprune_update)
    #score.f1 = np.mean(f1_rf_strut_noprune_update)
    #score.auc = np.mean(auc_rf_strut_noprune_update)
    #write_score(scores_file, score)
    #
    #score.algo = "strut noprune update"
    #score.score_rate = np.mean(score_rf_strut_noprune_noupdate)
    #score.tpr = np.mean(tpr_rf_strut_noprune_noupdate)
    #score.fpr = np.mean(fpr_rf_strut_noprune_noupdate)
    #score.f1 = np.mean(f1_rf_strut_noprune_noupdate)
    #score.auc = np.mean(auc_rf_strut_noprune_noupdate)
    #write_score(scores_file, score)
    
    # =============================================================================
    #       
    # =============================================================================

print('Scores :')
print('    Source : ', np.mean(score_rf))

print('    Target : ',np.mean(score_rf_target))
print('    Ser : ',np.mean(score_rf_ser))
print('    Ser no red : ',np.mean(score_rf_ser_no_red))
print('    Ser no ext : ',np.mean(score_rf_ser_no_ext))
print('    Ser no ext cond : ',np.mean(score_rf_ser_ext_cond))
print('    Ser no red no ext : ',np.mean(score_rf_ser_no_red_no_ext))
print('    Ser no red no ext cond: ',np.mean(score_rf_ser_no_red_no_ext_cond))
print('    Strut : ',np.mean(score_rf_strut))
print('    Strut adapt: ',np.mean(score_rf_strut_adapt))
print('    Strut noprune update : ',np.mean(score_rf_strut_noprune_update))
print('    Strut noprune noupdate: ',np.mean(score_rf_strut_noprune_noupdate))


print('ROC AUC :')
print('    Source : ',np.mean(auc_rf))
print('    Target : ',np.mean(auc_rf_target))
print('    Ser : ',np.mean(auc_rf_ser))
print('    Ser no red : ',np.mean(auc_rf_ser_no_red))
print('    Ser no ext : ',np.mean(auc_rf_ser_no_ext))
print('    Ser no ext cond : ',np.mean(auc_rf_ser_ext_cond))
print('    Ser no red no ext : ',np.mean(auc_rf_ser_no_red_no_ext))
print('    Ser no red no ext cond: ',np.mean(auc_rf_ser_no_red_no_ext_cond))
print('    Strut : ',np.mean(auc_rf_strut))
print('    Strut adapt : ',np.mean(auc_rf_strut_adapt))
print('    Strut noprune update : ',np.mean(auc_rf_strut_noprune_update))
print('    Strut noprune noupdate: ',np.mean(auc_rf_strut_noprune_noupdate))

print('F1 score :')
print('    Source : ',np.mean(f1_rf))
print('    Target : ',np.mean(f1_rf_target))
print('    Ser : ',np.mean(f1_rf_ser))
print('    Ser no red : ',np.mean(f1_rf_ser_no_red))
print('    Ser no ext : ',np.mean(f1_rf_ser_no_ext))
print('    Ser no ext cond : ',np.mean(f1_rf_ser_ext_cond))
print('    Ser no red no ext : ',np.mean(f1_rf_ser_no_red_no_ext))
print('    Ser no red no ext cond: ',np.mean(f1_rf_ser_no_red_no_ext_cond))
print('    Strut : ',np.mean(f1_rf_strut))
print('    Strut adapt : ',np.mean(f1_rf_strut_adapt))
print('    Strut noprune update : ',np.mean(f1_rf_strut_noprune_update))
print('    Strut noprune noupdate: ',np.mean(f1_rf_strut_noprune_noupdate))

#    print('TPR/FPR :')
#    print('    Source TPR : ',sum((rf.predict(X_target_test) == 1)*(Y_target_test == 1) ) / sum(Y_target_test == 1))
#    print('    Source FPR : ',sum((rf.predict(X_target_test) == 1)*(Y_target_test == 0) ) / sum(Y_target_test == 0))
#    
#    print('    Target TPR : ',sum((rf_target.predict(X_target_test) == 1)*(Y_target_test == 1) ) / sum(Y_target_test == 1))
#    print('    Target FPR : ',sum((rf_target.predict(X_target_test) == 1)*(Y_target_test == 0) ) / sum(Y_target_test == 0))
#    
#    print('    Ser TPR : ',sum((rf_ser.predict(X_target_test) == 1)*(Y_target_test == 1) ) / sum(Y_target_test == 1))
#    print('    Ser FPR : ',sum((rf_ser.predict(X_target_test) == 1)*(Y_target_test == 0) ) / sum(Y_target_test == 0))
#    
#    print('    Ser no red TPR : ',sum((rf_ser_no_red.predict(X_target_test) == 1)*(Y_target_test == 1) ) / sum(Y_target_test == 1))
#    print('    Ser no red FPR : ',sum((rf_ser_no_red.predict(X_target_test) == 1)*(Y_target_test == 0) ) / sum(Y_target_test == 0))
#    

# =============================================================================
#         kernel densities tests : 
# =============================================================================
#    from scipy import stats
#    from sklearn import cluster
#    
#    
#    X0 = a[1].values[Y_source == 0,:-1]
#    X1 = a[1].values[Y_source == 1,:-1]
#
#    ker0 = stats.gaussian_kde(X0.T)
#    ker1 = stats.gaussian_kde(X1.T)
#    
#    u0 = ker0.resample(250).T
#    u1 = ker1.resample(250).T
#    
#    z = np.zeros(500)
#    z[:250] = 1
#    
#    X = np.concatenate((u0,u1))
#    
#    spectral = cluster.SpectralClustering(
#        n_clusters=16, eigen_solver='arpack',
#        affinity="nearest_neighbors")
#    
#    spectral.fit(a[1].values[:,:-1])
#    
#    u = pd.DataFrame(np.concatenate((X,np.array([z]).T), axis = 1), columns = [1,2,3,"cluster"])
#    u = pd.DataFrame(np.concatenate((a[1].values[:,:-1],np.array([spectral.labels_]).T), axis = 1), columns = [1,2,3,"cluster"])
#
#
#    sns.pairplot( a[1], hue="cluster")
#    sns.pairplot( u, hue="cluster")
    


