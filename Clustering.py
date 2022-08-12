import numpy as np
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, DBSCAN, OPTICS
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import *
from kneed import KneeLocator
from Outputting import write_results_to_file
from Clust_Vis import *

# Clustering class with various clustering methods
class Clustering:
     def __init__(self, pointCloud, pcd_choice):
          self.pcd = pointCloud
          if (pcd_choice == "1"): self.type = "raw"
          elif (pcd_choice == "2"): self.type = "cldCmp"
          elif (pcd_choice == "3"): self.type = "pnet++"

          self.vis = Clust_Vis()
     
     # k means clustering method --> clusters a dataset into k (given) clusters
     def k_means_clustering(self, k):
          x = self.pcd
       
          print("\n------------------k means---------------------")
          n_init = 10
          kmeans = KMeans(n_clusters=k, n_init=n_init).fit(x) # number of clusters (k)
          y_km = kmeans.predict(x)

          arrResults = ["*************K-MEANS Parameters*************", "k:" + str(k), "n_init:" + str(n_init)]
          self.write_results(arrResults)
          
          unique_labels = np.unique(y_km)
          self.get_information(y_km, x, unique_labels)
          # self.vis.vis_k_means(unique_labels, kmeans.cluster_centers_, y_km, x)

          return unique_labels, y_km, "_kmeans"

      # OPTIC
     def optics_clustering(self):
          print("***************OPTICS CLUSTERING***************")
          X = self.pcd
          min_samp = 10
          xi = 0.05
          min_cluster_sz = 0.05
          max_e = 10
          print("starting optics method")
          clust = OPTICS(min_samples=min_samp, xi=xi, min_cluster_size=min_cluster_sz, max_eps=max_e).fit(X)
          y_op = clust.fit_predict(X)
          print("finished optics method")

          arrResults = ["*************OPTICS Parameters*************", "min_samples: " + str(min_samp), "min_cluster_size:" + str(min_cluster_sz), "xi:" + str(xi), "max_eps:" + str(max_e)]
          self.write_results(arrResults)
          
          unique_labels = np.unique(y_op)
          self.get_information(y_op, X, unique_labels)
          # self.vis.vis_OPTICS(X, clust.reachability_[clust.ordering_], y_op)
          
          return unique_labels, y_op, "_OPTICS"

     # DBSCAN 
     def dbscan_clustering(self):
          print("***************DBSCAN CLUSTERING***************")
          print("starting dbscan_clustering")
          X = self.pcd

          min_samples_ = 36 # for cloud compare with 15 features
          min_samples_ = 8 # for raw point cloud
          e = self.calculateElbow(min_samples_)

          arrResults = ["*************DBSCAN Parameters*************", "min_samples:"+str(min_samples_), "e:"+str(e)]
          self.write_results(arrResults)
          
          db = DBSCAN(eps=e, min_samples=min_samples_).fit(X)
          y_db = db.fit_predict(X)
          
          core_samples_mask = np.zeros_like(y_db, dtype=bool)
          core_samples_mask[db.core_sample_indices_] = True
          
          unique_labels = np.unique(y_db)
          self.get_information(y_db, X, unique_labels)
          # self.vis(X, y_db, db, core_samples_mask)

          return unique_labels, y_db, "_DBSCAN"

     def calculateElbow(self, n):
           # FIND OPTIMAL EPSILON VALUE: use elbow point detection method 
          nearest_neighbors = NearestNeighbors(n_neighbors=n)
          neighbors = nearest_neighbors.fit(self.pcd)
          distances, _ = neighbors.kneighbors(self.pcd)
          distances = np.sort(distances[:,4], axis=0)
          
          # IDENTIFY ELBOW POINT:
          print("Identify Elbow Point:")
          i = np.arange(len(distances))
          knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
          knee.plot_knee()
          eps = distances[knee.knee]
          # self.vis.vis_elbow_method(distances)
          return eps

     # MeanShift
     def mean_shift_clustering(self):
          print("***************MEAN-SHIFT CLUSTERING***************")
          X = self.pcd
          bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
          ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(X)
          y_ms = ms.predict(X)

          self.write_results(["*************MEAN-SHIFT Parameters*************", "quantile: 0.2", "n_samples: 500", "bin_seeding: True"])
          unique_labels = np.unique(y_ms)
          self.get_information(y_ms, X, unique_labels)
          # self.vis.vis_mean_shift(X, y_ms, ms)

          return unique_labels, y_ms, "_meanshift"

     def write_results(self, arrResults):
          for r in arrResults:
               write_results_to_file(r)

     def get_information(self, labels, X, unique_labels):
          print("Unique Labels:", unique_labels)
          no_clusters = len(np.unique(labels))
          no_noise = np.sum(np.array(labels) == -1, axis=0)
          clust = 'Estimated no. of clusters: %d' % no_clusters
          noise = 'Estimated no. of noise points: %d' % no_noise
          print(clust)
          print(noise)

          sil_score = silhouette_score(X, labels)
          db_index = davies_bouldin_score(X, labels)
          print("Silhouette Coefficient: %0.3f" % sil_score)
          print("Davies Bouldin Score: %0.3f" % db_index)
          
          arrResults = ["*************Clustering Metrics*************", "Silhouette Score:"+str(sil_score), "Davies Bouldin Index:"+str(db_index), clust, noise]
          self.write_results(arrResults)