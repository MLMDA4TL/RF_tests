import time
import sys
import numpy as np
import sklearn.ensemble as skl_ens
from sklearn.tree import export_graphviz
from collections import namedtuple

sys.path.insert(0, "../ser_strut/")
sys.path.insert(0, "../data_mngmt/")
sys.path.insert(0, "../utils/")
import data
# from transfTree import SER_RF
from SER_rec import SER_RF
from STRUT import STRUT_RF
from transfTree import fusionDecisionTree
from utils import error_rate, write_score


# =======================================================
#   Parameters
# =======================================================
scores_file = "../../output/scores_strut2_strut_ser_mix.csv"
# =======================================================
#   Data
# =======================================================
nb_trees = [1,10,50]
max_depths = [2,5,10,15,None]
for DATASET in ['wine']:
    print("Loading data : ", DATASET)
    load_data = getattr(data, "load_{}".format(DATASET))
    X_source, X_target_005, X_target_095, y_source, y_target_005, y_target_095 = load_data()
    for NB_TREE in nb_trees:
        for MAX_DEPTH in max_depths:
            for i in range(3):
                print(NB_TREE,MAX_DEPTH)
                # SCORES
                score = namedtuple("Score", "algo max_depth nb_tree data error_rate time")
                score.max_depth = MAX_DEPTH
                score.nb_tree = NB_TREE
                score.data = DATASET 
                if MAX_DEPTH is None:
                    score.max_depth = "None"

                # RF Source
                rf_source = skl_ens.RandomForestClassifier(
                    n_estimators=NB_TREE,
                    oob_score=True,
                    max_depth=MAX_DEPTH)
                rf_source.fit(X_source, y_source)
                rf_source_score_target = rf_source.score(X_target_095, y_target_095)
                score.algo = "source"
                score.error_rate = error_rate(rf_source_score_target)
                score.time = 0
                write_score(scores_file, score)

                # RF Target
                rf_target = skl_ens.RandomForestClassifier(n_estimators=NB_TREE,
                                                           oob_score=True,
                                                           class_weight=None)
                rf_target.fit(X_target_005, y_target_005)
                rf_target_score_target = rf_target.score(X_target_095, y_target_095)
                score.algo = "target"
                score.error_rate = error_rate(rf_target_score_target)
                score.time = 0
                write_score(scores_file, score)

                # RF SER
                start_time = time.time()
                rf_ser = SER_RF(rf_source, X_target_005, y_target_005)
                duration_ser = int(time.time() - start_time)
                rf_source_SER_score = rf_ser.score(X_target_095, y_target_095)
                score.algo = "ser"
                score.error_rate = error_rate(rf_source_SER_score)
                score.time = duration_ser
                write_score(scores_file, score)

                # RF STRUT
                start_time = time.time()
                rf_strut = STRUT_RF(rf_source, X_target_005, y_target_005,clean_unreachable_subtrees=True)
                duration_strut = int(time.time() - start_time)
                rf_source_STRUT_score = rf_strut.score(X_target_095, y_target_095)
                score.algo = "strut"
                score.error_rate = error_rate(rf_source_STRUT_score)
                score.time = duration_strut
                write_score(scores_file, score)

                # RF STRUT2
                start_time = time.time()
                rf_strut2 = STRUT_RF(rf_source, X_target_005, y_target_005,clean_unreachable_subtrees=False)
                duration_strut2 = int(time.time() - start_time)
                rf_source_STRUT2_score = rf_strut2.score(X_target_095, y_target_095)
                score.algo = "strut2"
                score.error_rate = error_rate(rf_source_STRUT2_score)
                score.time = duration_strut2
                write_score(scores_file, score)

                # MIX
                rf_mix = skl_ens.RandomForestClassifier(n_estimators=100, oob_score=True)
                rf_mix.fit(X_source, y_source)
                for i, dtree in enumerate(rf_strut.estimators_):
                    rf_mix.estimators_[i] = rf_strut.estimators_[i]
                for i, dtree in enumerate(rf_ser.estimators_):
                    rf_mix.estimators_[i + 50] = rf_ser.estimators_[i]
                rf_source_MIX_score = rf_mix.score(X_target_095, y_target_095) 
                score.algo = "mix"
                score.error_rate = error_rate(rf_source_MIX_score)
                score.time = 0
                write_score(scores_file, score)
                