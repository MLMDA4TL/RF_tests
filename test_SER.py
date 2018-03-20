import pandas as pd
import numpy as np
import sklearn.ensemble as skl_ens
from sklearn.tree import export_graphviz

from transfTree_rec import SER
from transfTree_rec import SER_RF
from STRUT import STRUT
from transfTree_rec import fusionDecisionTree


df = pd.read_csv("data/letter/letter-recognition.data.txt",
                 sep=',', header=None)
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
label = {letter: i for i, letter in enumerate(letters)}
df.iloc[:, 0] = [label[l] for l in df.iloc[:, 0]]

X_source = df[df[9] < np.median(df[9])]
X_target = df[df[9] >= np.median(df[9])]
X_target_005 = X_target.iloc[
    0: int(X_target.shape[0] * 0.05), list(range(1, len(X_source.columns)))]
y_target_005 = X_target.iloc[0: int(X_target.shape[0] * 0.05), 0]
X_target_095 = X_target.iloc[
    int(X_target.shape[0] * 0.05):, list(range(1, len(X_source.columns)))]
y_target_095 = X_target.iloc[int(X_target.shape[0] * 0.05):, 0]

rf_source = skl_ens.RandomForestClassifier(n_estimators=50, oob_score=True, max_depth = 10)
rf_source.fit(X_source[list(range(1, len(X_source.columns)))], X_source[0])

rf_target = skl_ens.RandomForestClassifier(n_estimators=50, oob_score=True, max_depth = 10)
rf_target.fit(X_target_005, y_target_005)

dtree0 = rf_source.estimators_[0]
dtree1 = rf_source.estimators_[1]

print('source only :',rf_source.score(X_target_095,y_target_095))
print('target only ',rf_target.score(X_target_095,y_target_095))

#==============================================================================
#  TEST 1 Arbre : 
#==============================================================================
#print("noeuds dtree0 : ", dtree0.tree_.node_count)
#
#export_graphviz(dtree0, "output/dtree_origin.dot")
#
#def updateValues(tree,values):
#    d = tree.__getstate__()
#    d['values'] = values
#    d['nodes']['n_node_samples'] = np.sum( values, axis = -1).reshape(-1)
#    d['nodes']['weighted_n_node_samples'] = np.sum( values, axis = -1).reshape(-1)
#    tree.__setstate__(d)
#   
#target_values = np.zeros((dtree0.tree_.node_count, 1, dtree0.tree_.value.shape[-1]))
#sparseM = dtree0.decision_path(X_target_005.values)
#
#for iy in range(target_values.shape[-1]):
#    target_values[:, 0, iy] = np.sum( sparseM.toarray()[np.where(y_target_005 == iy)[0]], axis=0)
#
#updateValues(dtree0.tree_, target_values)
#
#export_graphviz(dtree0, "output/dtree_origin_target.dot")
#
#SER(0,dtree0, X_target_005.values, y_target_005.values)
#
#print("noeuds après SER: ", dtree0.tree_.node_count)
#print(dtree0.predict(X_target_095.values))
#print(dtree0.score(X_target_095,y_target_095))
#
#export_graphviz(dtree0, "output/dtree_origin_rec.dot")

#==============================================================================
# 
#==============================================================================

rf_SER = SER_RF(rf_source, X_target_005.values, y_target_005.values)
print('SER :', rf_SER.score(X_target_095,y_target_095))
#print('La construction de SER implique que le score de l arbre devient :a'+str(dtree_SER_0.score(X_target_005,y_target_005))+'sur les donnéées d apprentissage')

#export_graphviz(dtree0, "output/dtree_0_SER.dot")
#export_graphviz(dtree1, "output/dtree_1_SER.dot")

#dtree_fusion = fusionDecisionTree(dtree0, 1, dtree1)
#export_graphviz(dtree_fusion, "output/dtree_fusion.dot")

#dtree_STRUT = STRUT(dtree0, 0, X_target_005.values, y_target_005.values)

# for i, dtree in enumerate(rf_source.estimators_):
    # print("Tree ", i)
