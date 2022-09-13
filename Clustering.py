import numpy as np
from sklearn.cluster import KMeans, MeanShift, DBSCAN, OPTICS
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import *
from kneed import KneeLocator
from Outputting import write_results_to_file
import pptk

class Clustering:
     '''This class is responsible for partitioning a point cloud into clusters using four different clustering algorithms'''
     def __init__(self, pointCloud):
          '''Initialise class variables
          args:
               pointCloud: the point cloud to cluster, stored in a NumPy array'''
          self.pcd = pointCloud
          
     def k_means_clustering(self, k, n_init=10):
          '''Cluster point cloud using k-means clustering method
          arg: 
               k: the number of clusters to partition the point cloud into
               n_init: number of iterations of k-means (default = 10)
          returns:
               unique_labels: an array containg the cluster indexes
               y_km: the produced clusters'''
          print("\n------------------k means---------------------")
          x = self.pcd
          kmeans = KMeans(n_clusters=k, n_init=n_init).fit(x) # number of clusters (k)
          y_km = kmeans.predict(x)

          self.write_results(["*************K-MEANS Parameters*************", "k:" + str(k), "n_init:" + str(n_init)])
          unique_labels = self.get_information(y_km, x)

          return unique_labels, y_km

     def optics_clustering(self, min_samp=10, xi=0.05, min_cluster_sz=25, max_e=100):
          '''Cluster point cloud using OPTICS clustering method
          arg: 
               min_samp:
               xi:
               min_cluster_size: (default=25)
               max_e:
          returns:
               unique_labels: an array containg the cluster indexes
               y_op: the produced clusters
               A suffix corresponding to the name of the clustering method
          '''
          print("***************OPTICS CLUSTERING***************")
          X = self.pcd
          clust = OPTICS(min_samples=min_samp, xi=xi, min_cluster_size=min_cluster_sz, max_eps=max_e).fit(X)  
          y_op = clust.fit_predict(X)

          self.write_results(["*************OPTICS Parameters*************", "min_samples: " + str(min_samp), "min_cluster_size:" + str(min_cluster_sz), "xi:" + str(xi), "max_eps:" + str(max_e)])
          unique_labels = self.get_information(y_op, X)
          
          return unique_labels, y_op

     def dbscan_clustering(self, min_samples_=6):
          '''Cluster point cloud using DBSCAN clustering method
          arg: 
               k: the number of clusters to partition the point cloud into
               min_samples: the minimum number of points per cluster
          returns:
               unique_labels: an array containg the cluster indexes
               y_db: the produced clusters
               A suffix corresponding to the name of the clustering method
          '''
          print("***************DBSCAN CLUSTERING***************")
          X = self.pcd
          e = self.calculateElbow(min_samples_)
          self.write_results(["*************DBSCAN Parameters*************", "min_samples:"+str(min_samples_), "e:"+str(e).replace('.', ',')])
          
          db = DBSCAN(eps=e, min_samples=min_samples_).fit(X)
          y_db = db.fit_predict(X)
          
          core_samples_mask = np.zeros_like(y_db, dtype=bool)
          core_samples_mask[db.core_sample_indices_] = True
          unique_labels = self.get_information(y_db, X)

          return unique_labels, y_db

     def calculateElbow(self, n):
          '''Calculate the value of e for DBSCAN
          args: 
               n: min_samples
          returns:
               eps: the value of e for DBSCAN
          '''
          # Find optimal epsilon value: use elbow point detection method 
          nearest_neighbors = NearestNeighbors(n_neighbors=n)
          neighbors = nearest_neighbors.fit(self.pcd)
          distances, _ = neighbors.kneighbors(self.pcd)
          distances = np.sort(distances[:,4], axis=0)
          
          # Identify Elbow Point
          i = np.arange(len(distances))
          knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
          knee.plot_knee()
          eps = distances[knee.knee]
          return eps

     def mean_shift_clustering(self, bandwidth=1):
          '''Cluster point cloud using mean-shift clustering method
          arg: 
               bandwidth: (default = 1)
          returns:
               unique_labels: an array containg the cluster indexes
               y_ms: the produced clusters
               A suffix corresponding to the name of the clustering method
          '''
          print("***************MEAN-SHIFT CLUSTERING***************")
          X = self.pcd
          ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(X)
          y_ms = ms.predict(X)

          self.write_results(["*************MEAN-SHIFT Parameters*************", "bandwidth:" + str(bandwidth).replace('.', ','),  "bin_seeding: True"])
          unique_labels = self.get_information(y_ms, X)

          return unique_labels, y_ms

     def write_results(self, arrResults):
          '''Write results to a file
          args: 
               arrResults: the array of strings to write to file'''
          for r in arrResults:
               write_results_to_file(r)

     def get_information(self, labels, X):
          '''Method to get information about produced clusters
          args:
               labels: the cluster label of each cluster
               X: the point Cloud
               unique_labels: an array of unique cluster labels
          '''
          unique_labels = np.unique(labels)
          print("Unique Labels:", unique_labels)
          no_clusters = len(np.unique(labels))
          no_noise = np.sum(np.array(labels) == -1, axis=0)
          clust = 'Estimated no. of clusters: %d' % no_clusters
          noise = 'Estimated no. of noise points: %d' % no_noise
          print(clust, '\n', noise)

          sil_score = silhouette_score(X, labels)
          db_index = davies_bouldin_score(X, labels)
          print("Silhouette Coefficient: %0.3f" % sil_score)
          print("Davies Bouldin Score: %0.3f" % db_index)
          
          self.write_results(["*************Clustering Metrics*************", "Silhouette Score:"+str(sil_score).replace('.', ','), "Davies Bouldin Index:"+str(db_index).replace('.', ','), clust, noise])
          self.visualise_clustering(labels, X)
          return unique_labels
     
     def visualise_clustering(self, labels, x):
          '''Visualise Clustered Point Cloud with pptk
          args:
               labels: the label associated with each cluster
               x: the point cloud to visualise'''
          pptk.viewer(x[:,0:3], labels.flatten())