import pandas as pd
import numpy as np
from Generator import ClusterPoints
SEPARATOR_FOR_PARAMETER_NAMES = "|"
COMMENTS_SYMBOL = "#"



def apply_drift(clusters, cluster_feature_speed, min_coordinate, max_coordinate):
	def elastic_collision(cluster,min_coordinate,max_coordinate):
		cluster.centroid[cluster.centroid >= max_coordinate] = max_coordinate
		cluster.centroid[cluster.centroid <= min_coordinate] = min_coordinate

	for cluster_id, feature_speed in cluster_feature_speed.items():
		for feature, speed in feature_speed.items():
			clusters[cluster_id].centroid[feature] += speed
		elastic_collision(clusters[cluster_id],min_coordinate,max_coordinate)

def apply_density_change(clusters, cluster_feature_std):
	for cluster_id, feature_std in cluster_feature_std.items():
		for feature, std in feature_std.items():
			clusters[cluster_id].radii[feature] = std

def create_new_clusters(clusters,list_parameters_cluster_creation):
	for parameter in list_parameters_cluster_creation:
		new_cluster = ClusterPoints(**parameter)
		clusters.append(new_cluster)

def delete_clusters(clusters, clusters_id):
	for cluster_id in clusters_id:
		clusters.pop(cluster_id)

def change_cluster_weight(clusters, clusters_weights):
	for cluster_id, weight in clusters_weights.items():
		clusters[cluster_id].weight = weight

def loose_feature(clusters,clusters_features):
	for cluster_id, features in clusters_features.items():
		for feature in features:
			if clusters[cluster_id].projected_dims[feature]:
				clusters[cluster_id].projected_dims[feature] = 0
				clusters[cluster_id].nb_projected_dims -= 1
				clusters[cluster_id].nb_not_projected_dims += 1

def gain_feature(clusters,clusters_features):
	for cluster_id, features in clusters_features.items():
		for feature in features:
			if not clusters[cluster_id].projected_dims[feature]:
				clusters[cluster_id].projected_dims[feature] = 1
				clusters[cluster_id].nb_not_projected_dims -= 1
				clusters[cluster_id].nb_projected_dims += 1
