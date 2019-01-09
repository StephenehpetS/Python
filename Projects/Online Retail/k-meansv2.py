import pandas as pd
# matrix math
import numpy as np
# graphing2D
import matplotlib.pyplot as plt
# graphing 3D
from mpl_toolkits import mplot3d as plt2
# graphing animation
import matplotlib.animation as animation
# for feature dimensional reduction Method
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage
from MLinAction.Project2.prepareData import prepareData, standard


def load_dataset(path):
  rfmTable = prepareData(path)
  data = standard(rfmTable)
  return data


# euclidian distance between two data points
def euclidian(x, y):
  return np.linalg.norm(x - y)


# finding the optimimal number of clusters
def elbow_plot(dataset, maxK=10):
  """
      parameters:
      - data: pandas DataFrame (data to be fitted)
      - maxK (default = 10): integer (maximum number of clusters with which to run k-means)
      - seed_centroids (default = None ): float (initial value of centroids for k-means)
  """
  # Elbow method
  K = range(1, 30)
  meandistortions = []
  for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(dataset)
    meandistortions.append(sum(np.min(
      cdist(dataset, kmeans.cluster_centers_, 'euclidean'), axis=1)) / dataset.shape[0])
  plt.plot(K, meandistortions, 'bx-')
  plt.xlabel('Number of clusters')
  plt.ylabel('Average Distortions')
  plt.title('Elbow Analysis')
  plt.show()

  # parameters:
  """
  k: the number of clusters (required)
  epsilon: the minimum error to be used in the stop condition (optional,defauld==0)
  Distance: the method is used to calculate distance and has the return:
  history of centroids- store prevoiusely processed centroids including the optimal one 
  """


def kmeans(k, dataset, epsilon=0, distance='euclidean'):
  # list to store previous centroides or history
  history_centroids = []
  # set the distance calculation method
  # get the number of instances and number of features
  num_instances, num_features = dataset.shape
  # define k cetroids choosen randomly
  centroids = dataset[np.random.randint(0, num_instances - 1, size=k)]
  # set the protopype to our previous centroids to show progress over time
  history_centroids.append(centroids)
  # to keep track of centroid at every iteration
  centroids_old = np.zeros(centroids.shape)
  # to store clusters
  belongs_to = np.zeros((num_instances, 1))
  norm = euclidian(centroids, centroids_old)
  total_num_iter = 0

  while norm > epsilon:
    total_num_iter += 1
    print('distance between centroids', norm)
    norm = euclidian(centroids, centroids_old)
    # for each instance in the dataset
    for index_instance, instance in enumerate(dataset):
      # define a distance vector of size k
      dist_vector = np.zeros((k, 1))

      # for each Centroid
      for index_centroid, centroid in enumerate(centroids):
        # compute the distance between x and centroid
        dist_vector[index_centroid] = euclidian(centroid, instance)
      # find the smallest distance and assign that distance to a cluster
      belongs_to[index_instance, 0] = np.argmin(dist_vector)
    temp_centroids = np.zeros((k, num_features))

    # for each cluster k of them
    for index in range(len(centroids)):
      # get all the points assigned to a cluster
      instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
      # find the mean of those instances, this is our new centroid
      centroid = np.mean(dataset[instances_close], axis=0)
      # add our new centroid to our temporary list
      temp_centroids[index, :] = centroid
      # assign the previous centroids to centroids old
    centroids_old = centroids;
    # set the new list to prototypes list
    centroids = temp_centroids
    # add our calculated centroids to  our history for plotting
    history_centroids.append(temp_centroids)

  # print the value of norm
  print('distance between centroids', norm)
  # print the value of total_num_iter
  print('Total Iterations done to find optimal Centroids ', total_num_iter)
  # return the calculated centroids, hostory of them all,all the data points belonging to each cluster
  return centroids, history_centroids, belongs_to


# plotting algorithm for our dataset and our centroids
def plot(dataset, history_centroids, belongs_to):
  # three colors for each centroid
  colors = ['red', 'green', 'yellow']
  # three labels for each centroid
  labels = ['Claster 1', 'Cluster 2', 'Cluster 3']
  # split our graph by its axis and actual plot
  fig, ax = plt.subplots();
  # for each point in our set
  for index in range(3):
    instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
    # assign each data point in that cluster a color and plot
    for instance_index in instances_close:
      # ax.plot(dataset[instance_index][0],dataset[instance_index][1],colors[index])
      plt.scatter(dataset[instance_index][0], dataset[instance_index][1], s=100, c=colors[index], label=labels[index],
                  alpha=0.5, marker='+')
      plt.title('Clusters of students')
      plt.xlabel('PCA 1')
      plt.ylabel('PCA 2')

  # let's also log the history of centroids calculated via training
  history_points = []
  # for each centroid ever calculated
  for index, centroids in enumerate(history_centroids):
    # print them all out
    for inner, item in enumerate(centroids):
      if index == 0:
        history_points.append(ax.plot(item[0], item[1], 'bo')[0])
      else:
        history_points[inner].set_data(item[0], item[1])

        plt.pause(0.8)


# normalize the dataset -attributes will have to be normilize 0 to 1
# def normilize_data(dataset):
# for each instance in the dataset

# lets do a PCA for feature dimensional reduction
def dimensional_reduction(dataset):
  pca = PCA(n_components=2)
  dataset_questions_pca = pca.fit_transform(dataset)

  return dataset_questions_pca


# main file goes here
def main():
  # load dataset
  data = load_dataset("c:\\Online_Retail.xlsx")

  # implementing the Hierarchical Clustering
  X = np.array(data)  # change dataframe into ndarray

  dataset_questions_pca = dimensional_reduction(X)

  # display number of instances and features
  instances, features = X.shape
  print('Before appliying PCA')
  print('Number of instances: ', instances)
  print('Number of features: ', features)
  print('After appliying PCA')

  instances, features = dataset_questions_pca.shape
  print('Number of instances: ', instances)
  print('Number of features: ', features)

  # set number of clusters
  k = 3
  episelon = 0

  elbow_plot(dataset_questions_pca)
  # train the model on the dataset
  centroids, history_centroids, belongs_to = kmeans(k, dataset_questions_pca, episelon, 'euclidean')

  # plot the results
  plot(dataset_questions_pca, history_centroids, belongs_to)


# run the program now
main()






