import pandas as pd
import numpy as np
SEPARATOR_FOR_PARAMETER_NAMES = "|"
COMMENTS_SYMBOL = "#"

class ClusterPoints:
	def __init__(self,
				 weight,
				 dimensionality,
				 min_projected_dim,
				 max_projected_dim,
				 min_coordinate,
				 max_coordinate,
				 min_projected_dim_var,
				 max_projected_dim_var,
				 class_label,
				 ):
		# set dimensionality, weight, boundary and class label
		self.dimensionality = dimensionality
		self.weight = weight
		self.min_coordinate = min_coordinate
		self.max_coordinate = max_coordinate
		self.class_label = class_label
		# Draw initial centroids locations
		self.centroid = self.generate_uniform_value(max_coordinate, min_coordinate)
		# Draw clusters radii
		self.radii = self.generate_uniform_value(max_projected_dim_var, min_projected_dim_var)
		self.radii = np.sqrt(self.radii)
		# Draw projected dimensions and not projected dimensions
		self.nb_projected_dims = np.random.randint(min_projected_dim,max_projected_dim+1)
		self.nb_not_projected_dims = self.dimensionality - self.nb_projected_dims
		self.projected_dims = np.hstack((np.ones(self.nb_projected_dims), np.zeros(self.dimensionality - self.nb_projected_dims)))
		np.random.shuffle(self.projected_dims)
		print(self.centroid)
	
	def generate_uniform_value(self, min_value, max_value, size = None):
		if size is None: size = self.dimensionality
		return np.random.random(size)*(max_value - min_value) + min_value

	def generate_point(self):
		point_coordinates = np.random.randn(self.dimensionality) * self.radii + self.centroid
		point_coordinates[np.logical_not(self.projected_dims)] = self.generate_uniform_value(self.min_coordinate, self.max_coordinate, self.nb_not_projected_dims)
		return point_coordinates
	
class StreamGenerator:
	def __init__(self, 
				 number_points,
				 weights,
				 dimensionality,
				 min_projected_dim,
				 max_projected_dim,
				 min_coordinate,
				 max_coordinate,
				 min_projected_dim_var,
				 max_projected_dim_var,
				 class_labels = None,
				 ):
		self.number_points = number_points
		self.dimensionality = dimensionality
		self.min_projected_dim = min_projected_dim 
		self.max_projected_dim = max_projected_dim
		self.min_coordinate = min_coordinate
		self.max_coordinate = max_coordinate
		self.min_projected_dim_var = min_projected_dim_var
		self.max_projected_dim_var = max_projected_dim_var
		self.weights = np.asarray(weights)
		self.clusters = []
		if class_labels is None: 
			class_labels = range(len(self.weights))
		self.class_labels = class_labels
		for i,weight in enumerate(self.weights):
			cluster  = ClusterPoints(weight,
									 dimensionality,
									 min_projected_dim,
									 max_projected_dim,
									 min_coordinate,
									 max_coordinate,
									 min_projected_dim_var,
									 max_projected_dim_var,
									 class_labels[i],
									 )
			self.clusters.append(cluster)
		self.compute_probabilities_draw_cluster()
		

	def run(self):
		for i in range(self.number_points):
			self.compute_probabilities_draw_cluster()
			cluster_id = np.random.choice(range(len(self.clusters)),1,p=self.probability_draw_cluster)[0]
			new_point = self.clusters[cluster_id].generate_point()
			yield np.hstack((new_point,self.clusters[cluster_id].class_label))
			

	def compute_probabilities_draw_cluster(self):
		weights = np.asarray([cluster.weight for cluster in self.clusters])
		self.probability_draw_cluster = weights / weights.sum()
		#self.p.loc[self.i] = self.probability_draw_cluster
		#self.i +=1

	def get_full_dataset(self, size):
		self.number_points = size
		stream_df = []
		stream_ss = []
		for i,c in enumerate(self.run()):
		    stream_df.append(c)#loc[i] = c 
		    stream_ss.append(self.clusters[int(c[-1])].projected_dims.copy())#loc[i] =
		stream_df = pd.DataFrame(stream_df,columns= list(range(self.dimensionality))+["cluster"])
		stream_ss = pd.DataFrame(stream_ss,columns= list(range(self.dimensionality)))
		return stream_ss, stream_df

	def get_file_name(self):
		return "D_"+str(self.dimensionality)+"_C_"+str(self.weights.size)+"_N_"+str(self.number_points)
