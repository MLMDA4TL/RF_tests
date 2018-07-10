#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 02:44:03 2018

@author: mounir
"""
import glob
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


#==============================================================================
# 
#==============================================================================

   
def depth(tree,f):
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
    
    return np.amax(depth_array(t,np.linspace(0,n-1,n).astype(int)))


def max_depth_rf(rf):
    p = 0
    for e in rf.estimators_:
        if max_depth_dTree(e) > p :
            p = max_depth_dTree(e)
    return p

def predict_proba_without_weight(rf,X):

    P = np.zeros((X.shape[0],rf.n_estimators))
    
    S = np.zeros((X.shape[0],rf.n_classes_))
    for u in range(rf.n_estimators):
        P[:,u] = rf.estimators_[u].predict(X)

    for i in range(rf.n_classes_):
        filt = (P == i)
        S[:,i] = np.sum(1*filt/rf.n_estimators,axis=1)
    
    return S

# =============================================================================
# 
# =============================================================================
path_data = 'outputs/'
path_results = 'outputs/results/'
name_results = 'results.csv'

N_EXP_START = 0
N_EXP = 18
N_DS = 10
N_REP = 10



N_EST = 50
MAX_DEPTH = 15

name_results = 'results_'+str(N_EST)+'Trees_depth'+str(MAX_DEPTH)+'_Exps'+str(N_EXP)+'_NDataSets'+str(N_DS)+'_runs'+str(N_REP)+'.csv'

bootstrap = True
moving_thresh = False
beta = 1


results = pd.DataFrame(columns=["Exp","Dataset","Rep","Alg","TPR","FPR","F-score","ROC AUC","av prec"])

# =============================================================================
# 
# =============================================================================


for e in range(N_EXP_START, N_EXP):
    print("\n\n\n\n EXP n ° "+str(e)+"\n\n\n\n")
    for k in range(N_DS):
        
        if len(glob.glob(path_data+'*Exp'+str(e)+'*Rep'+str(k)+'*.csv')) != 0:

            file_data_source = glob.glob(path_data+'*Train_source*Exp'+str(e)+'*Rep'+str(k)+'*.csv')[0]
            file_data_target_train = glob.glob(path_data+'*Train_target*Exp'+str(e)+'*Rep'+str(k)+'*.csv')[0]
            file_data_target_test = glob.glob(path_data+'*Test_target*Exp'+str(e)+'*Rep'+str(k)+'*.csv')[0]
            
            data = pd.read_csv(file_data_source)
            
            X_source = data.values[:,:-1]
            Y_source = data.values[:,-1]
            data = pd.read_csv(file_data_target_train)
            
            X_target_train = data.values[:,:-1]
            Y_target_train = data.values[:,-1]
            
            data = pd.read_csv(file_data_target_test)
            
            X_target_test = data.values[:,:-1]
            Y_target_test = data.values[:,-1]
            
            tpr_rf = np.zeros(N_REP)
            fpr_rf = np.zeros(N_REP)
            tpr_targ = np.zeros(N_REP)
            fpr_targ = np.zeros(N_REP)
            tpr_ser = np.zeros(N_REP)
            fpr_ser = np.zeros(N_REP)
            tpr_ser_no_red = np.zeros(N_REP)
            fpr_ser_no_red = np.zeros(N_REP)
            tpr_strut = np.zeros(N_REP)
            fpr_strut = np.zeros(N_REP)
            fscore = np.zeros(N_REP)
            fscore_targ = np.zeros(N_REP)
            fscore_ser =  np.zeros(N_REP)
            fscore_ser_no_red = np.zeros(N_REP)
            fscore_strut = np.zeros(N_REP)
            auc_score = np.zeros(N_REP)
            auc_score_targ = np.zeros(N_REP)
            auc_score_ser = np.zeros(N_REP)
            auc_score_ser_no_red = np.zeros(N_REP)
            auc_score_strut = np.zeros(N_REP)
            av_score = np.zeros(N_REP)
            av_score_targ = np.zeros(N_REP)
            av_score_ser = np.zeros(N_REP)
            av_score_ser_no_red = np.zeros(N_REP)
            av_score_strut = np.zeros(N_REP)
            
            for i in range(N_REP):

                
                rf = skl_ens.RandomForestClassifier(n_estimators = N_EST,max_depth = MAX_DEPTH, bootstrap = bootstrap  )
                
                rf_th = skl_ens.RandomForestClassifier(n_estimators = N_EST, bootstrap = bootstrap  )
                rf_th.fit(X_target_test,Y_target_test)
                
                rf_target = skl_ens.RandomForestClassifier(n_estimators = N_EST, bootstrap = bootstrap )
                
                rf.fit(X_source,Y_source)
                rf_target.fit(X_target_train,Y_target_train)
                
                rf_ser = SER_rec.SER_RF(rf, X_target_train, Y_target_train)
                rf_ser_no_red = SER_rec.SER_RF(rf, X_target_train, Y_target_train, no_red_on_cl = [1])
                rf_strut = STRUT.STRUT_RF(rf, X_target_train, Y_target_train,1)
                
                y_rf = rf.predict(X_target_test) 
                tpr_rf[i] = sum((y_rf == 1)*(Y_target_test == 1) ) / sum(Y_target_test == 1)
                fpr_rf[i] = sum((y_rf == 1)*(Y_target_test == 0) ) / sum(Y_target_test == 0)
                
                
                y_targ_p = rf_target.predict_proba(X_target_test) 
                y_ser_p = rf_ser.predict_proba(X_target_test) 
                y_ser_no_red_p = rf_ser_no_red.predict_proba(X_target_test) 
                #y_ser_no_red_p = predict_proba_without_weight(rf_ser_no_red,X_target_test)
                y_strut_p = rf_strut.predict_proba(X_target_test)
                
                f,p,t = sklearn.metrics.roc_curve(Y_target_test, y_targ_p[:,1])
                thresh = t[1:-1][np.argmin( np.abs(p[1:-1] - tpr_rf[i]) )]
                tpr_targ[i] = sum((y_targ_p[:,1] > thresh)*(Y_target_test == 1) ) / sum(Y_target_test == 1)
                fpr_targ[i] = sum((y_targ_p[:,1] > thresh)*(Y_target_test == 0) ) / sum(Y_target_test == 0)
                
                y_targ = rf_target.predict(X_target_test) 
                
                f,p,t = sklearn.metrics.roc_curve(Y_target_test,  y_ser_p[:,1])
                thresh = t[1:-1][np.argmin( np.abs(p[1:-1] - tpr_rf[i]) )]
                tpr_ser[i] = sum((y_ser_p[:,1]  > thresh)*(Y_target_test == 1) ) / sum(Y_target_test == 1)
                fpr_ser[i] = sum((y_ser_p[:,1]  > thresh)*(Y_target_test == 0) ) / sum(Y_target_test == 0)
                
                y_ser = rf_ser.predict(X_target_test) 
                
                f,p,t = sklearn.metrics.roc_curve(Y_target_test, y_ser_no_red_p[:,1])
                thresh = t[1:-1][np.argmin( np.abs(p[1:-1] - tpr_rf[i]) )]
                tpr_ser_no_red[i] = sum((y_ser_no_red_p[:,1]  > thresh)*(Y_target_test == 1) ) / sum(Y_target_test == 1)
                fpr_ser_no_red[i] = sum((y_ser_no_red_p[:,1]  > thresh)*(Y_target_test == 0) ) / sum(Y_target_test == 0)
                
                y_ser_no_red = rf_ser_no_red.predict(X_target_test) 
                
                f,p,t = sklearn.metrics.roc_curve(Y_target_test, y_strut_p[:,1])
                thresh = t[1:-1][np.argmin( np.abs(p[1:-1] - tpr_rf[i]) )]
                tpr_strut[i] = sum((y_strut_p[:,1] > thresh)*(Y_target_test == 1) ) / sum(Y_target_test == 1)
                fpr_strut[i] = sum((y_strut_p[:,1] > thresh)*(Y_target_test == 0) ) / sum(Y_target_test == 0)
                
                y_strut = rf_strut.predict(X_target_test) 
                
                if moving_thresh != 0:
                     
                    tpr_targ[i] = sum((y_targ == 1)*(Y_target_test == 1) ) / sum(Y_target_test == 1)
                    tpr_ser[i] = sum((y_ser == 1)*(Y_target_test == 1) ) / sum(Y_target_test == 1)
                    tpr_ser_no_red[i] = sum((y_ser_no_red == 1)*(Y_target_test == 1) ) / sum(Y_target_test == 1)
                    tpr_strut[i] = sum((y_strut == 1)*(Y_target_test == 1) ) / sum(Y_target_test == 1)
                    #tpr_rf = sum((y_rf == 1)*(Y_target_test == 1) ) / sum(Y_target_test == 1)
                    
                    
                    
                    fpr_targ[i] = sum((y_targ == 1)*(Y_target_test == 0) ) / sum(Y_target_test == 0)
                    fpr_ser[i] = sum((y_ser == 1)*(Y_target_test == 0) ) / sum(Y_target_test == 0)
                    fpr_ser_no_red[i] = sum((y_ser_no_red == 1)*(Y_target_test == 0) ) / sum(Y_target_test == 0)
                    fpr_strut[i] = sum((y_strut == 1)*(Y_target_test == 0) ) / sum(Y_target_test == 0)
                
                fscore[i] = sklearn.metrics.fbeta_score(Y_target_test, y_rf,beta)
                fscore_targ[i] = sklearn.metrics.fbeta_score(Y_target_test, y_targ,beta)
                fscore_ser[i] = sklearn.metrics.fbeta_score(Y_target_test, y_ser,beta)
                fscore_ser_no_red[i] = sklearn.metrics.fbeta_score(Y_target_test, y_ser_no_red,beta)
                fscore_strut[i] = sklearn.metrics.fbeta_score(Y_target_test, y_strut,beta)
            
                auc_score[i] = sklearn.metrics.roc_auc_score(Y_target_test, rf.predict_proba(X_target_test)[:,1])
                auc_score_targ[i] = sklearn.metrics.roc_auc_score(Y_target_test, rf_target.predict_proba(X_target_test)[:,1])
                auc_score_ser[i] = sklearn.metrics.roc_auc_score(Y_target_test, rf_ser.predict_proba(X_target_test)[:,1])
                auc_score_ser_no_red[i] = sklearn.metrics.roc_auc_score(Y_target_test, rf_ser_no_red.predict_proba(X_target_test)[:,1])
                auc_score_strut[i] = sklearn.metrics.roc_auc_score(Y_target_test, rf_strut.predict_proba(X_target_test)[:,1])
            
                av_score[i] = sklearn.metrics.average_precision_score(Y_target_test, rf.predict_proba(X_target_test)[:,1])
                av_score_targ[i] = sklearn.metrics.average_precision_score(Y_target_test, rf_target.predict_proba(X_target_test)[:,1])
                av_score_ser[i] = sklearn.metrics.average_precision_score(Y_target_test, rf_ser.predict_proba(X_target_test)[:,1])
                av_score_ser_no_red[i] = sklearn.metrics.average_precision_score(Y_target_test, rf_ser_no_red.predict_proba(X_target_test)[:,1])
                av_score_strut[i] = sklearn.metrics.average_precision_score(Y_target_test, rf_strut.predict_proba(X_target_test)[:,1])    
            
                l = pd.Series({"Exp":e,"Dataset":k,"Rep":i,"Alg":"Source","TPR":tpr_rf[i],"FPR":fpr_rf[i],"F-score":fscore[i],"ROC AUC":auc_score[i],"av prec":av_score[i]})
                results = results.append(l,ignore_index=True)
                l = pd.Series({"Exp":e,"Dataset":k,"Rep":i,"Alg":"TargetOnly","TPR":tpr_targ[i],"FPR":fpr_targ[i],"F-score":fscore_targ[i],"ROC AUC":auc_score_targ[i],"av prec":av_score_targ[i]})
                results = results.append(l,ignore_index=True)
                l = pd.Series({"Exp":e,"Dataset":k,"Rep":i,"Alg":"Ser","TPR":tpr_ser[i],"FPR":fpr_ser[i],"F-score":fscore_ser[i],"ROC AUC":auc_score_ser[i],"av prec":av_score_ser[i]})
                results = results.append(l,ignore_index=True)
                l = pd.Series({"Exp":e,"Dataset":k,"Rep":i,"Alg":"SerNoRed1","TPR":tpr_ser_no_red[i],"FPR":fpr_ser_no_red[i],"F-score":fscore_ser_no_red[i],"ROC AUC":auc_score_ser_no_red[i],"av prec":av_score_ser_no_red[i]})
                results = results.append(l,ignore_index=True)
                l = pd.Series({"Exp":e,"Dataset":k,"Rep":i,"Alg":"Strut","TPR":tpr_strut[i],"FPR":fpr_strut[i],"F-score":fscore_strut[i],"ROC AUC":auc_score_strut[i],"av prec":av_score_strut[i]})
                results = results.append(l,ignore_index=True)
            #    l = pd.Series({"Exp":e,"Dataset":k,"Rep":i,"Alg":"Strut_no_red1","TPR":tpr_rf[i],"FPR":tpr_rf[i],"F-score":fscore[i],"ROC AUC":auc_score[i],"av prec":av_score[i]})
            #    results = results.append(l,ignore_index=True)
                
#        else:
#            print('Exp '+str(e)+' non trouvee')
                


results.to_csv(path_results + name_results)
