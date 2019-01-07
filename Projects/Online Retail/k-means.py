from sklearn import metrics

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from MLinAction.Project2.prepareData import prepareData, standard
from MLinAction.Project2.silhouette import silhouette, Elbow

data = prepareData("c:\\Online_Retail.xlsx")

# k-Means
km = KMeans(n_clusters=5)
km.fit(data)
data['cluster'] = km.labels_
cluster_center = km.cluster_centers_
centers = data.groupby("cluster").mean().reset_index()
colors = np.array(['red', 'green', 'blue', 'yellow', 'purple', ])
plt.rcParams['font.size'] = 14
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data["R"], data["F"], data["M"], c=colors[data['cluster']])
ax.get_label()
plt.show()
# end of k-Means


# Calinski-Harabaz Index
for i in range(2, 10):
  km = KMeans(n_clusters=i, random_state=9).fit_predict(PCA(2).fit_transform(data))
  X, y = PCA(2).fit_transform(data), km
  silhouette(X, y, i)
  # print(metrics.calinski_harabaz_score(X, y))
# End of Calinski-Harabaz Index
