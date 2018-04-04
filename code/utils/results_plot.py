import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
max_depth = 50
results = pd.read_csv("../../output/scores_strut2_strut_ser_mix.csv",sep=",",header=0)
results["max_depth"][results.max_depth == "None"] = max_depth
results["max_depth"] = results["max_depth"].values.astype(float)
results["max_depth"][np.isnan(results.max_depth.values)] = max_depth

plt.subplot(3,1,1)
plt.ylim([40,80])
results_1 = results[results["nb_tree"] == 1]
sns.swarmplot(x="max_depth", y="error_rate(perc)",hue="algo",data=results_1,alpha=0.7)

plt.subplot(3,1,2)
plt.ylim([40,80])
results_10 = results[results["nb_tree"] == 10]
sns.swarmplot(x="max_depth", y="error_rate(perc)",hue="algo",data=results_10,alpha=0.7)

plt.subplot(3,1,3)
plt.ylim([40,80])
results_50 = results[results["nb_tree"] == 50]
sns.swarmplot(x="max_depth", y="error_rate(perc)",hue="algo",data=results_50,alpha=0.7)

plt.show()

