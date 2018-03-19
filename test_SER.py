import sklearn.ensemble as skl_ens
from sklearn.tree import export_graphviz

import data
from transfTree import SER_RF
from STRUT import STRUT
from transfTree import fusionDecisionTree

# =======================================================
#   Parameters
# =======================================================
NB_TREE = 50
APPLY_SER = 1

# =======================================================
#   Data
# =======================================================
X_source, X_target_005, X_target_095, y_source, y_target_005, y_target_095 = data.load_letter()

# =======================================================
#   CLF initialisation
# =======================================================
rf_source = skl_ens.RandomForestClassifier(
    n_estimators=NB_TREE,
    oob_score=True)
rf_source.fit(X_source, y_source)
rf_source_score_target = rf_source.score(X_target_095, y_target_095)
print("Error rate de rf_source sur data target : ",
      round(1 - rf_source_score_target, 2))

rf_target = skl_ens.RandomForestClassifier(n_estimators=NB_TREE,
                                           oob_score=True,
                                           class_weight=None)
rf_target.fit(X_target_005, y_target_005)
rf_target_score_target = rf_target.score(X_target_095, y_target_095)
print("Error rate de rf_target(5%) sur data target(95%) : ",
      round(1 - rf_target_score_target, 2))

# =======================================================
#   Tests
# =======================================================
# dtree0 = rf_source.estimators_[0]
# dtree1 = rf_source.estimators_[1]

# dtree_test = skl_tree.DecisionTreeClassifier()
# dtree_test.fit(X_target_005, y_target_005)

# print("noeuds dtree0 : ", dtree0.tree_.node_count)
# print("noeuds dtree1 : ", dtree1.tree_.node_count)

# export_graphviz(dtree0, "output/dtree_0.dot")
# export_graphviz(dtree1, "output/dtree_1.dot")

# dtree_SER_0 = SER(dtree0, X_target_005.values, y_target_005.values)
# dtree_SER_1 = SER(dtree1, X_target_005.values, y_target_005.values)

# export_graphviz(dtree0, "output/dtree_0_SER.dot")
# export_graphviz(dtree1, "output/dtree_1_SER.dot")

# dtree_fusion = fusionDecisionTree(dtree0, 1, dtree1)
# export_graphviz(dtree_fusion, "output/dtree_fusion.dot")

# dtree_STRUT = STRUT(dtree0, 0, X_target_005.values, y_target_005.values)

# =======================================================
#   SER algorithm
# =======================================================
if APPLY_SER:
    SER_RF(rf_source, X_target_005, y_target_005)
    # now every tree of rf_source has been modified
    rf_source_SER_score = rf_source.score(X_target_095, y_target_095)
    print("Error rate de rf_source_SER sur data target(95%) : ",
          round(1 - rf_source_SER_score, 2))
