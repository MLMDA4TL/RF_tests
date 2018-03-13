import pandas as pd
import numpy as np
import sklearn.ensemble as skl_ens
from sklearn.tree import export_graphviz

from transfTree import SER
from STRUT import STRUT
from transfTree import fusionDecisionTree


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

rf_source = skl_ens.RandomForestClassifier(n_estimators=50, oob_score=True)
rf_source.fit(X_source[list(range(1, len(X_source.columns)))], X_source[0])
rf_source.score(X_target[list(range(1, len(X_source.columns)))], X_target[0])

rf_target = skl_ens.RandomForestClassifier(n_estimators=50, oob_score=True)

rf_target.fit(X_target_005, y_target_005)

rf_target.score(X_target_095, y_target_095)

dtree0 = rf_source.estimators_[0]
dtree1 = rf_source.estimators_[1]

print("noeuds dtree0 : ", dtree0.tree_.node_count)
print("noeuds dtree1 : ", dtree1.tree_.node_count)

export_graphviz(dtree0, "output/dtree_0.dot")
export_graphviz(dtree1, "output/dtree_1.dot")

dtree_SER_0 = SER(dtree0, X_target_005.values, y_target_005.values)
dtree_SER_1 = SER(dtree1, X_target_005.values, y_target_005.values)

export_graphviz(dtree0, "output/dtree_0_SER.dot")
export_graphviz(dtree1, "output/dtree_1_SER.dot")

dtree_fusion = fusionDecisionTree(dtree0, 1, dtree1)
export_graphviz(dtree_fusion, "output/dtree_fusion.dot")

dtree_STRUT = STRUT(dtree0, 0, X_target_005.values, y_target_005.values)

# for i, dtree in enumerate(rf_source.estimators_):
    # print("Tree ", i)
