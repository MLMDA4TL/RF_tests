import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
max_depth = 50
results = pd.read_csv("../../output/scores.csv",sep=",",header=0)
results["max_depth"][results.max_depth == "None"] = max_depth
results["max_depth"] = results["max_depth"].values.astype(float)
results["max_depth"][np.isnan(results.max_depth.values)] = max_depth

sns.swarmplot(x="max_depth", y="error_rate(perc)",hue="algo",data=results)
plt.show()

