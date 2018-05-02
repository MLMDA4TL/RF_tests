import numpy as np
from sklearn import tree
import copy
from collections import OrderedDict 
from common_functions import get_list_split_phi
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import accuracy_score
from sklearn.utils.linear_assignment_ import linear_assignment 
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
		intersections.append(inter)
		unions.append(thresholds_a.size + thresholds_b.size - inter)
	return np.asarray(intersections).sum() *1./ np.asarray(unions).sum()

def CE(y_A,y_B):
    confusion_matrix = pd.crosstab(y_A,y_B)
    best_A_B_couples = linear_assignment(-confusion_matrix)
    return sum([confusion_matrix.iloc[couple[0],couple[1]] for couple in best_A_B_couples])*1./sum(confusion_matrix.sum(0))

def CE_based_comparison(dt_a, dt_b, X):
	leafs_ids_a = dt_a.apply(X)
	leafs_ids_b = dt_b.apply(X)
	return CE(leafs_ids_a,leafs_ids_b)

def accuracy_based_distance(dt_a, dt_b, X):
	y_a = dt_a.predict(X)
	y_b = dt_b.predict(X)
	return accuracy_score(y_a, y_b)

def compare_trees_from_forest(rf, similarity_function, **params):
	similarity_matrix = np.ones((len(rf.estimators_),len(rf.estimators_)))
	for i in range(len(rf.estimators_)-1):
		for j in range(i+1, len(rf.estimators_)):
			dt_a = rf.estimators_[i]
			dt_b = rf.estimators_[j]
			similarity_matrix[i,j] = similarity_function(dt_a,dt_b, **params)
			similarity_matrix[j,i] = similarity_matrix[i,j]
	return similarity_matrix

if __name__ == "__main__":
  # Build a dataset
  import matplotlib.pyplot as plt
  from sklearn.tree import DecisionTreeClassifier,export_graphviz
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.cluster import SpectralClustering
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
  rf = RandomForestClassifier(max_depth=5,n_estimators=100)
  rf.fit(X, Y)
  similarity_matrix_splits = compare_trees_from_forest(rf, splits_based_distance, gamma=10)
  similarity_matrix_accuracy = compare_trees_from_forest(rf, accuracy_based_distance,X=X)
  similarity_matrix_ce = compare_trees_from_forest(rf, CE_based_comparison, X=X)
  sns.clustermap(similarity_matrix_splits,method="ward")
  plt.show()
  sns.clustermap(similarity_matrix_accuracy,method="ward")
  plt.show()
  sns.clustermap(similarity_matrix_ce,method="ward")
  plt.show()

  sc = SpectralClustering(n_clusters=10)
  clusters_accuracy = sc.fit_predict(similarity_matrix_accuracy)

  sc = SpectralClustering(n_clusters=10)
  clusters_ce = sc.fit_predict(similarity_matrix_ce)

  sc = SpectralClustering(n_clusters=10)
  clusters_splits = sc.fit_predict(similarity_matrix_splits)

  plt.imshow(pd.crosstab(clusters_splits,clusters_ce))
  plt.show()
  plt.imshow(pd.crosstab(clusters_splits,clusters_accuracy))
  plt.show()
  plt.imshow(pd.crosstab(clusters_accuracy,clusters_ce))
  plt.show()
