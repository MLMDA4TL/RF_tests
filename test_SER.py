import sklearn.ensemble as skl_ens
from sklearn.tree import export_graphviz

import data
from SER_rec import SER_RF
from SER_rec import SER
from STRUT import STRUT_RF
from SER_rec import fusionDecisionTree
from utils import error_rate

from data_viz import dot_treatment
# =======================================================
#   Parameters
# =======================================================
NB_TREE = 50
MAX_DEPTH = 10
APPLY_SER = 0
APPLY_STRUT = 0
APPLY_MIX = 0

# =======================================================
#   Data
# =======================================================
X_source, X_target_005, X_target_095, y_source, y_target_005, y_target_095 = data.load_letter()

# =======================================================
#   CLF initialisation
# =======================================================


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

# =======================================================
#   Tests
# =======================================================
dtree0 = rf_source.estimators_[0]
#dtree1 = rf_source.estimators_[1]
#
#dtree_test = skl_tree.DecisionTreeClassifier()
##dtree_test.fit(X_target_005, y_target_005)
#
#print("noeuds dtree0 : ", dtree0.tree_.node_count)
#print("noeuds dtree1 : ", dtree1.tree_.node_count)
#
export_graphviz(dtree0, "output/dtree_0.dot")
#dot_treatment('output/dtree_0.dot','dtree_0_v2',dtree0,weighted=True, filt = 0.01)
#export_graphviz(dtree1, "output/dtree_1.dot")
#
SER(0,dtree0, X_target_005, y_target_005)
#dtree_SER_1 = SER(0,dtree1, X_target_005, y_target_005)
#
export_graphviz(dtree0, "output/dtree_0_SER.dot")
#dot_treatment('output/dtree_0_SER.dot','dtree_0_SER_v2',dtree0,weighted=True, filt = 0.01)
##export_graphviz(dtree1, "output/dtree_1_SER.dot")
#
##dtree_fusion = fusionDecisionTree(dtree0, 1, dtree1)
##export_graphviz(dtree_fusion, "output/dtree_fusion.dot")
#
##dtree_STRUT = STRUT(dtree0, 0, X_target_005.values, y_target_005.values)

# =======================================================
#   SER algorithm
# =======================================================


if APPLY_SER:
    rf_ser = SER_RF(rf_source, X_target_005, y_target_005, bootstrap_ = False)
    # nb: rf_source is not modified (deep copy inside function)
    rf_source_SER_score = rf_ser.score(X_target_095, y_target_095)
    print("Error rate de rf_ser sur data target(95%) : ",
          error_rate(rf_source_SER_score))

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

