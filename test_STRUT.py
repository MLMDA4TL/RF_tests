import pandas as pd
import numpy as np
import sklearn.tree as skl_tree
import sklearn.ensemble as skl_ens
from sklearn.tree import export_graphviz

from transfTree import SER
from STRUT import STRUT


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
rf_source_score_target = rf_source.score(X_target[list(range(1, len(X_source.columns)))], X_target[0])
print("score rf_source on target095: ", round(1-rf_source_score_target, 2))

rf_target = skl_ens.RandomForestClassifier(n_estimators=50, oob_score=True,
        class_weight='balanced')

rf_target.fit(X_target_005, y_target_005)
rf_target_score_target = rf_target.score(X_target_095, y_target_095)

print("score rf_target005 on target095: ", round(1-rf_target_score_target, 2))

dtree0 = rf_source.estimators_[0]
dtree1 = rf_source.estimators_[1]

export_graphviz(dtree0, "output/dtree0_beforeSTRUT.dot")
export_graphviz(dtree1, "output/dtree1_beforeSTRUT.dot")

STRUT(dtree0, 0, X_target_005.values, y_target_005.values)
STRUT(dtree1, 0, X_target_005.values, y_target_005.values)

export_graphviz(dtree0, "output/dtree0_afterSTRUT.dot")
export_graphviz(dtree1, "output/dtree1_afterSTRUT.dot")
