from Generator import ClusterPoints,StreamGenerator
import matplotlib.pyplot as plt
import seaborn as sns

sg = StreamGenerator(number_points=100,
					 weights=[0.05,0.05,0.45,0.45],
					 dimensionality=3,
					 min_projected_dim=3,
					 max_projected_dim=3,
					 min_coordinate=-2,
					 max_coordinate=2,
					 min_projected_dim_var=0.1,
					 max_projected_dim_var=0.3,
					 class_labels=[0,0,1,1],
					 )
a = sg.get_full_dataset(200)
sns.pairplot(a[1],hue="cluster")
plt.show()

from Changes import apply_drift

cluster_feature_speed = {1:{0:0.1, 2:0.1}, 3: {1:0.1, 2:0.1}}
apply_drift(sg.clusters, 
			cluster_feature_speed,
			sg.min_coordinate,
			sg.max_coordinate)

b = sg.get_full_dataset(200)
sns.pairplot(b[1],hue="cluster")
plt.show()