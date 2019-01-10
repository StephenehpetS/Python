from sklearn import metrics

from matplotlib import pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from MLinAction.Project2.prepareData import prepareData, standard
from MLinAction.Project2.silhouette import silhouette

# np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation

rfmTable = prepareData("c:\\Online_Retail.xlsx")
data = standard(rfmTable)


def fancy_dendrogram(*args, **kwargs):
  max_d = kwargs.pop('max_d', None)
  if max_d and 'color_threshold' not in kwargs:
    kwargs['color_threshold'] = max_d
  annotate_above = kwargs.pop('annotate_above', 0)

  ddata = dendrogram(*args, **kwargs)

  if not kwargs.get('no_plot', False):
    plt.title('Hierarchical Clustering Dendrogram (truncated)')
    plt.xlabel('sample index or (cluster size)')
    plt.ylabel('distance')
    for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
      x = 0.5 * sum(i[1:3])
      y = d[1]
      if y > annotate_above:
        plt.plot(x, y, 'o', c=c)
        plt.annotate("%.3g" % y, (x, y), xytext=(0, -5), textcoords='offset points', va='top', ha='center')
    if max_d:
      plt.axhline(y=max_d, c='k')
  return ddata


# implementing the Hierarchical Clustering
# change dataframe into ndarray
X = np.array(rfmTable)
Z = linkage(X, 'ward')

# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
fancy_dendrogram(
  Z,
  truncate_mode='lastp',
  p=12,
  leaf_rotation=90.,
  leaf_font_size=12.,
  show_contracted=True,
  annotate_above=10,  # useful in small plots so annotations don't overlap
)
plt.show()


# plt.figure(figsize=(10, 8))
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='prism')  # plot points with cluster dependent colors
# plt.show()

for i in range(2, 10):
  y = fcluster(Z, i, criterion='maxclust')
  silhouette(X, y, i)
  # print(metrics.calinski_harabaz_score(X, y))
