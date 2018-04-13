import numpy as np
import sys
import sklearn.ensemble as skl_ens
from sklearn.tree import export_graphviz
sys.path.insert(0, "../ser_strut/")
sys.path.insert(0, "../data_mngmt/")
sys.path.insert(0, "../utils/")
import data
from SER_rec import SER_RF
from SER_rec import SER
from STRUT import STRUT_RF
from SER_rec import fusionDecisionTree
from utils import error_rate

# =======================================================
#   Parameters
# =======================================================
NB_TREE = 20
APPLY_SER = 1
APPLY_STRUT = 0
APPLY_MIX = 0

SIZE_TEST = 1

# =======================================================
#   Data
# =======================================================

X_source, X_target_005, X_target_095, y_source, y_target_005, y_target_095 = data.load_letter()
#
#cl_to_reduce = 5
#th = 1
#
#for i in range(1):
#    for k,y in enumerate(y_target_005):
#        p = np.random.random()
#        if y == cl_to_reduce:
#            
#           ind = np.random.randint(0,y_target_095.size)
#           y_targ05 = y_target_005[k]
#           X_targ05 = X_target_005[k,:]
#           
#           y_target_005[k] = y_target_095[ind]
#           y_target_095[ind] = y_targ05
#           X_target_005[k,:] = X_target_095[ind,:]
#           X_target_095[ind,:] = X_targ05
       
#==============================================================================
# 
#==============================================================================


#==============================================================================
# 
#==============================================================================

def voyelle_cons(y):
    y_voy = np.zeros(y.size)
    for i,lab in enumerate(y):
        if ( lab == 0 ) or ( lab == 4 ) or ( lab == 8 ) or ( lab == 14 ) or ( lab == 20 ) or ( lab == 24 ) :
            y_voy[i] = 1
    return y_voy
            
            
#y_source = voyelle_cons(y_source)
#ind_source_0 = np.where(y_source == 0)[0]
#ind_source_1 = np.where(y_source == 1)[0]
#
#y_target_005 = voyelle_cons(y_target_005)
#ind_target_005_0 = np.where(y_target_005 == 0)[0]
#ind_target_005_1 = np.where(y_target_005 == 1)[0]
#
#y_target_095 = voyelle_cons(y_target_095)
#
#y_target = np.concatenate((y_source[ind_source_0],y_target_005[ind_target_005_1]))
#y_s = np.concatenate((y_source[ind_source_1],y_target_005[ind_target_005_0]))
#X_s = np.concatenate((X_source[ind_source_1],X_target_005[ind_target_005_0]))
#X_target = np.concatenate((X_source[ind_source_0],X_target_005[ind_target_005_1]))
#
#X_source = X_s
#y_source = y_s
#
#X_target_005 = X_target
#y_target_005 = y_target
#
#rf_isolation_source = skl_ens.IsolationForest(n_estimators=NB_TREE, contamination = np.sum(y_source)/y_source.size)
#rf_isolation_source.fit(X_source)
#
#y_pred = ( rf_isolation_source.predict(X_source) == 1 )
#
#print('Erreur isolation : ',1 - np.sum(y_pred == y_source)/y_pred.size)

# =======================================================
#   CLF initialisation
# =======================================================

MAX_DEPTH = 5


rf_source = skl_ens.RandomForestClassifier(n_estimators=NB_TREE, max_depth = MAX_DEPTH , oob_score=True)
rf_target = skl_ens.RandomForestClassifier(n_estimators=NB_TREE, max_depth = MAX_DEPTH , oob_score=True, class_weight=None)

rf_source.fit(X_source, y_source)
rf_source_score_target = rf_source.score(X_target_095, y_target_095)
print("Error rate de rf_source sur data target : ",
      error_rate(rf_source_score_target))


rf_target.fit(X_target_005, y_target_005)
rf_target_score_target = rf_target.score(X_target_095, y_target_095)
print("Error rate de rf_target(5%) sur data target(95%) : ",
      error_rate(rf_target_score_target))

#for i in range(SIZE_TEST):
#    print('Test n°', i)
#    
#    rf_source.fit(X_source, y_source)
#    rf_source_score_target = rf_source[i].score(X_target_095, y_target_095)
#    print("Error rate de rf_source sur data target : ",
#          error_rate(rf_source_score_target))
#    
#
#    rf_target.fit(X_target_005, y_target_005)
#    rf_target_score_target = rf_target.score(X_target_095, y_target_095)
#    print("Error rate de rf_target(5%) sur data target(95%) : ",
#          error_rate(rf_target_score_target))
#
#    scores_source[i] = rf_source_score_target
#    scores_targ[i] = rf_target_score_target 
#    
# =======================================================
#   Tests
# =======================================================
#dtree0 = rf_source.estimators_[0]
#dtree1 = rf_source.estimators_[1]
#
##dtree_test = skl_tree.DecisionTreeClassifier()
##dtree_test.fit(X_target_005, y_target_005)
#
#print("noeuds dtree0 : ", dtree0.tree_.node_count)
#print("noeuds dtree1 : ", dtree1.tree_.node_count)
#
##export_graphviz(dtree0, "output/dtree_0.dot")
##export_graphviz(dtree1, "output/dtree_1.dot")
#
#dtree_SER_0 = SER(0,dtree0, X_target_005, y_target_005)
#dtree_SER_1 = SER(0,dtree1, X_target_005, y_target_005)
#
##export_graphviz(dtree0, "output/dtree_0_SER.dot")
##export_graphviz(dtree1, "output/dtree_1_SER.dot")
#
##dtree_fusion = fusionDecisionTree(dtree0, 1, dtree1)
##export_graphviz(dtree_fusion, "output/dtree_fusion.dot")
#
##dtree_STRUT = STRUT(dtree0, 0, X_target_005.values, y_target_005.values)

# =======================================================
#   SER algorithm
# =======================================================

#==============================================================================
# 
#==============================================================================
if APPLY_SER:
    rf_ser = SER_RF(rf_source, X_target_005, y_target_005, bootstrap_ = False)
    rf_ser2 = SER_RF(rf_source, X_target_005, y_target_005, bootstrap_ = False, no_red = True, cl_no_red = [0,12,15,21])
    # nb: rf_source is not modified (deep copy inside function)
    rf_source_SER_score = rf_ser.score(X_target_095, y_target_095)
    print("Error rate de rf_ser sur data target(95%) : ",
          error_rate(rf_source_SER_score))
    export_graphviz(rf_ser.estimators_[0], "../../output/dtree_0_SER.dot")
    rf_source_SER_score2 = rf_ser2.score(X_target_095, y_target_095)
    print("Error rate de rf_ser no red sur data target(95%) : ",
          error_rate(rf_source_SER_score2))
    export_graphviz(rf_ser2.estimators_[0], "../../output/dtree_0_SER_condred.dot")
    
if APPLY_STRUT:
    rf_strut = STRUT_RF(rf_source, X_target_005, y_target_005)
    rf_source_STRUT_score = rf_strut.score(X_target_095, y_target_095)
    print("Error rate de rf_strut sur data target(95%) : ",
          error_rate(rf_source_STRUT_score))

if APPLY_MIX:
    rf_mix = skl_ens.RandomForestClassifier(n_estimators=100, oob_score=True)
    # fit to create estimatros_
    rf_mix.fit(X_source, y_source)
    for i, dtree in enumerate(rf_strut.estimators_):
        rf_mix.estimators_[i] = rf_strut.estimators_[i]
    for i, dtree in enumerate(rf_ser.estimators_):
        rf_mix.estimators_[i + 50] = rf_ser.estimators_[i]
    rf_source_MIX_score = rf_mix.score(X_target_095, y_target_095)
    print("Error rate de rf_mix sur data target(95%) : ",
          error_rate(rf_source_MIX_score))

#==============================================================================
#   TEST ON  SER
#==============================================================================
import matplotlib.pyplot as plt
from SER_rec import max_depth_rf

print('Prof source :', max_depth_rf(rf_source))

inds = np.zeros(rf_source.n_classes_,dtype =object)
sc_s = np.zeros(rf_source.n_classes_)
sc_targO = np.zeros(rf_source.n_classes_)
sc_ser = np.zeros(rf_source.n_classes_)
sc_ser2 = np.zeros(rf_source.n_classes_)

a,_ = np.histogram(y_target_005,bins= rf_source.n_classes_)
plt.bar(rf_source.classes_,a)

for i in range(rf_source.n_classes_):
    inds[i] = np.where( y_target_095 == i )[0]
    sc_ser2[i] = rf_ser2.score(X_target_095[inds[i]], y_target_095[inds[i]])
    sc_ser[i] = rf_ser.score(X_target_095[inds[i]], y_target_095[inds[i]])
    sc_s[i] = rf_source.score(X_target_095[inds[i]], y_target_095[inds[i]])
    sc_targO[i] = rf_target.score(X_target_095[inds[i]], y_target_095[inds[i]])

fig  = plt.figure(figsize = (6,9)) 
ax = fig.add_subplot(211) 
ax.plot(sc_s, label = 'Source')
ax.plot(sc_ser,label = 'OnlyTarget')
ax.plot(sc_targO,label = 'SER')

ind_c = np.where( sc_ser < sc_s )[0]
ax.scatter( ind_c , sc_ser[ind_c], c = 'r')

ax2 = fig.add_subplot(312) 
ax2.plot(sc_s, label = 'Source')
ax2.plot(sc_ser2, label = 'OnlyTarget')
ax2.plot(sc_targO, label = 'SER avoiding reduc.')

ind_c = np.where( sc_ser2 < sc_s )[0]
ax2.scatter( ind_c , sc_ser2[ind_c], c = 'r')

#ax3 = fig.add_subplot(313) 
#ax3.plot(sc_ser, 'r')
#ax3.plot(sc_ser2, 'blue')

#==============================================================================
#   TEST ON  STRUT
#==============================================================================
#
#sc_strut = np.zeros(rf_source.n_classes_)
#
#for i in range(rf_source.n_classes_):
#
#    sc_strut[i] = rf_strut.score(X_target_095[inds[i]], y_target_095[inds[i]])
#
#fig2  = plt.figure() 
#ax2 = fig2.add_subplot(111) 
#ax2.plot(sc_s)
#ax2.plot(sc_strut)
#ax2.plot(sc_targO)
#
#ind_c = np.where( sc_strut < sc_s )[0]
#ax2.scatter( ind_c , sc_strut[ind_c], c = 'r')