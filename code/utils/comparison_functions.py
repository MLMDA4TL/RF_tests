import numpy as np
from sklearn import tree
import copy
from collections import OrderedDict 
from common_functions import get_list_split_phi
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd


def group_splits_by_feature(list_splits):
	feature_list_splits = {}
	for id_split,split in list_splits.iteritems():
		feature = split[0]
		if feature not in feature_list_splits:
			feature_list_splits[feature] = OrderedDict()
		feature_list_splits[feature][id_split] = split[1]
	return feature_list_splits

def splits_based_distance(dt_a, dt_b, distance="manhattan", gamma=5,epsilon=None):
	list_splits_a = get_list_split_phi(dt_a)
	list_splits_b = get_list_split_phi(dt_b)
	gb_feature_a = group_splits_by_feature(list_splits_a)
	gb_feature_b = group_splits_by_feature(list_splits_b)
	full_features = list(set(gb_feature_a.keys()).intersection(set(gb_feature_b.keys())))
	intersections = []
	unions = []
	for feature in full_features:
		thresholds_a = np.asarray(gb_feature_a[feature].values()).reshape(-1, 1)
		thresholds_b = np.asarray(gb_feature_b[feature].values()).reshape(-1, 1)
		if epsilon is None:
			if gamma is None:
				gamma = 0.999		
			intra_a_distances = np.triu(pairwise_distances(thresholds_a,thresholds_a,metric=distance),1)
			intra_b_distances = np.triu(pairwise_distances(thresholds_b,thresholds_b,metric=distance),1)
			intra_distrib = np.asarray(list(intra_a_distances[intra_a_distances!=0])+list(intra_b_distances[intra_b_distances!=0]))
			epsilon = np.percentile(intra_distrib,gamma)
			#epsilon = gamma*min(intra_a_distances[intra_a_distances!=0].min(), intra_b_distances[intra_b_distances!=0].min())
		distances = pairwise_distances(thresholds_a,thresholds_b,metric=distance)
		axis=0
		if thresholds_a.size < thresholds_b.size:
			axis=1
		inter = (distances.min(axis=axis)<=epsilon).sum()
		print inter,thresholds_a.size, thresholds_b.size
		intersections.append(inter)
		unions.append(thresholds_a.size + thresholds_b.size - inter)
	return np.asarray(intersections).sum() *1./ np.asarray(unions).sum()





group_splits_by_feature(get_list_split_phi(clf))

if __name__ == "__main__":
  # Build a dataset
  import matplotlib.pyplot as plt
  from sklearn.tree import DecisionTreeClassifier,export_graphviz
  from sklearn.ensemble import RandomForestClassifier
  import seaborn as sns
  dataset_length = 1000
  D = 2
  X = np.random.randn(dataset_length,D)*0.1
  X[0:dataset_length//2,0] += 0.1
  X[0:dataset_length//2,0] += 0.2
  Y = np.ones(dataset_length)
  Y[0:dataset_length//2] *= 0

  X_test = np.random.randn(dataset_length,D)*0.1
  X_test[0:dataset_length//2,0] += 0.1
  X_test[0:dataset_length//2,0] += 0.2
  Y_test = np.ones(dataset_length)
  Y_test[0:dataset_length//2] *= 0
  # Train a Tree
  dt_a = DecisionTreeClassifier()
  dt_a = dt_a.fit(X, Y)
  # Train another Tree
  dt_b = DecisionTreeClassifier()
  dt_b = dt_b.fit(X_test, Y_test)
  splits_based_distance(dt_a, dt_b,gamma=5)

  #______ test forests
  rf = RandomForestClassifier(max_depth=5,n_estimators=10)
  rf.fit(X, Y)
  similarity_matrix = np.ones((len(rf.estimators_),len(rf.estimators_)))
  for i in range(len(rf.estimators_)-1):
    for j in range(i+1, len(rf.estimators_)):
      dt_a = rf.estimators_[i]
      dt_b = rf.estimators_[j]
      similarity_matrix[i,j] = splits_based_distance(dt_a,dt_b,gamma=10)
      similarity_matrix[j,i] = similarity_matrix[i,j]
  sns.clustermap(similarity_matrix)
  plt.show()

