import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
import faiss
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import *
from kneed import KneeLocator
from itertools import cycle
from Outputting import write_results_to_file, img_path

# Clustering class with various clustering methods
class Clustering:
     def __init__(self, pointCloud, pcd_with_truths, pcd_choice):
          self.pcd = pointCloud
          self.pcd_truth = pcd_with_truths
          if (pcd_choice == "1"): self.type = "raw"
          elif (pcd_choice == "2"): self.type = "cldCmp"
          elif (pcd_choice == "3"): self.type = "pnet++"
     
     # K-MEANS CLUSTERING USING FAISS LIBRARY - SPEEDS UP COMPUTATION
     def k_means_clustering_faiss(self, k, imageName):
      x = self.pcd
      t = self.pcd_truth
      print("starting faiss_k_means")
      # train:
      n_init = 10
      max_iter = 300
      kmeans = faiss.Kmeans(d=x.shape[1], k=k, niter=max_iter, nredo=n_init)
      kmeans.train(x.astype(np.float32))

      # predict:
      prediction = kmeans.index.search(x.astype(np.float32), 1)[1]
      y_km = prediction.flatten()
      print("finished faiss_k_means")
     
      # Visualise K-Means
      centroids = kmeans.centroids
      print("centroids:", centroids)
      unique_labels = np.unique(y_km)
      print("unique_labels:", unique_labels)
      
      for i in unique_labels:
          plt.scatter(x[y_km == i , 0] , x[y_km == i , 1] , label = i, marker='o', picker=True)
          print("plotting", i)
      plt.scatter(
          centroids[:, 0], centroids[:, 1],
          s=100, marker='*',
          c='red', edgecolor='black',
          label='centroids'
      )
      print("creating title")
      plt.title('K-Means Clustering')
      print("saving figure")
      plt.savefig('k_means_clusters_' + self.type + imageName + '.png') 
      plt.show()

     # k means clustering method --> clusters a dataset into k (given) clusters
     def k_means_clustering(self, k):
          x = self.pcd
          t = self.pcd_truth
       
          print("\n------------------k means---------------------")
          kmeans = KMeans(n_clusters=k, n_init=10) # number of clusters (k)
          kmeans.fit(x) # apply k means to dataset

          write_results_to_file("*************K-MEANS Parameters*************")
          write_results_to_file("K:" + str(k))
          write_results_to_file("n_init: 10")
          
          # Visualise K-Means
          y_km = kmeans.predict(x)
          print("y_km:", y_km)
          centroids = kmeans.cluster_centers_
          print("centroids:", centroids)
          
          t = self.pcd_truth
          unique_labels = np.unique(y_km)
          self.get_information(y_km, x, unique_labels, t)

          print("plotting graph")   
          for i in unique_labels:
               plt.scatter(x[y_km == i , 0] , x[y_km == i , 1] , label = i, marker='o', picker=True)
          plt.scatter(
               centroids[:, 0], centroids[:, 1],
               s=100, marker='*',
               c='red', edgecolor='black',
               label='centroids'
          )
          #plt.legend()
          plt.title('K-Means Clustering')
          plt.savefig(img_path + 'k_means_clusters.png') 
          plt.show()

          return unique_labels, y_km, t, "_kmeans"

      # OPTIC
     def optics_clustering(self):
          print("***************OPTICS CLUSTERING***************")
          X = self.pcd
          epsilon = 2.0
          min_samples = 3
          cluster_method = 'xi'
          metric = 'minkowski'
          print("starting optics method")
          clust = OPTICS(min_samples=4, xi=0.05, min_cluster_size=0.05).fit(X)
          print("finished optics method")

          write_results_to_file("*************OPTICS Parameters*************")
          write_results_to_file("min_samples: 4")
          write_results_to_file("min_cluster_size: 0.05")
          write_results_to_file("xi: 0.05")

          # Run the fit
          #clust = clust.fit(X)
          #print('labels:', labels)
          space = np.arange(len(X))
          reachability = clust.reachability_[clust.ordering_]
          labels = clust.labels_#[clust.ordering_]
          print('labels2:', labels)

          t = self.pcd_truth
          unique_labels = np.unique(clust.labels_)
          self.get_information(labels, X, unique_labels, t)

          # Reachability plot
          colors = ["g.", "r.", "b.", "y.", "c."]
          for klass, color in zip(range(0, 5), colors):
               Xk = space[labels == klass]
               Rk = reachability[labels == klass]
               plt.plot(Xk, Rk, color, alpha=0.3)
               plt.plot(space[labels == -1], reachability[labels == -1], "k.", alpha=0.3)
               plt.plot(space, np.full_like(space, 2.0, dtype=float), "k-", alpha=0.5)
               plt.plot(space, np.full_like(space, 0.5, dtype=float), "k-.", alpha=0.5)
               plt.title("Reachability (epsilon distance)")
               plt.ylabel("Reachability Plot")
          plt.savefig(img_path + "OPTICS_reachability.png")
          plt.show()
          
          # OPTICS
          colors = ["g.", "r.", "b.", "y.", "c."]
          for klass, color in zip(range(0, 5), colors):
               print("color", color)
               Xk = X[labels == klass]
               plt.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
               plt.plot(X[labels == -1, 0], X[labels == -1, 1], "k+", alpha=0.1)
               plt.title("Automatic Clustering\nOPTICS")
          plt.tight_layout()
          plt.savefig(img_path + "Optics_clusters1_" + self.type + '.png')
          plt.show()

          # visualise
          imgName = img_path + 'Optics_clusters_' + self.type + '.png'
          self.visualiseClusters("Optics Clustering", X, labels, imgName)

          return unique_labels, clust.labels_, t, "_OPTICS"
          
     # DBSCAN 
     def dbscan_clustering(self):
          print("***************DBSCAN CLUSTERING***************")
          print("starting dbscan_clustering")
          X = self.pcd

          min_samples_ = 36 # for cloud compare with 15 features
          min_samples_ = 8 # for raw point cloud
          e = self.calculateElbow(min_samples_)
          print("e=",e)

          write_results_to_file("*************DBSCAN Parameters*************")
          write_results_to_file("min_samples:" + str(min_samples_))
          write_results_to_file("e:" + str(e))
          
          db1 = DBSCAN(eps=e, min_samples=min_samples_)
          db = db1.fit(X)
          predict = db.fit_predict(X)
          print("labels:", db.labels_)
          
          core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
          core_samples_mask[db.core_sample_indices_] = True
          
          t = self.pcd_truth
          unique_labels = np.unique(db.labels_)
          self.get_information(db.labels_, X, unique_labels, t)

          # visualise
          imgName = img_path + 'DBSCAN_clusters_' + self.type + '.png'
          self.visualiseClusters("DBSCAN-Shift Clustering", X, db.labels_, imgName)
          
          # visualise 2
          imgName = img_path + 'DBSCAN_clusters2_' + self.type + '.png'
          self.visualiseClusters2("DBSCAN-Shift Clustering2", X, db.labels_, imgName, predict)
          
           # visualise 4
          imgName = img_path + "DBSCAN_clusters1_" + self.type + '.png'
          self.visualiseClusters4("DBSCAN Clustering 4", X, db.labels_, imgName, core_samples_mask)
          
          print("finished dbscan_clustering")

          return unique_labels, db.labels_, t, "_DBSCAN"

     def calculateElbow(self, n):
           # FIND OPTIMAL EPSILON VALUE: use elbow point detection method 
          df = self.pcd
          nearest_neighbors = NearestNeighbors(n_neighbors=n)
          neighbors = nearest_neighbors.fit(df)
          distances, indices = neighbors.kneighbors(df)
          distances = np.sort(distances[:,4], axis=0)
          plt.plot(distances)
          plt.xlabel("Points")
          plt.ylabel("Distance")
          plt.savefig(img_path + 'DBSCAN_Eps.png')
          plt.show()
          
          # IDENTIFY ELBOW POINT:
          print("Identify Elbow Point:")
          i = np.arange(len(distances))
          knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
          knee.plot_knee()
          plt.xlabel("Points")
          plt.ylabel("Distance")
          plt.show()
          plt.savefig(img_path + 'DBSCAN_elbow.png')
          eps = distances[knee.knee]
          return eps

     # MeanShift
     def mean_shift_clustering(self):
          print("***************MEAN-SHIFT CLUSTERING***************")
          X = self.pcd
          bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
          ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
          ms.fit(X)

          write_results_to_file("*************MEAN-SHIFT Parameters*************")
          write_results_to_file("quantile: 0.2")
          write_results_to_file("n_samples: 500")
          write_results_to_file("bin_seeding: True")

          t = self.pcd_truth
          unique_labels = np.unique(ms.labels_)
          self.get_information(ms.labels_, X, unique_labels, t)

          # visualise 1
          imgName = img_path + 'Mean-Shift_clusters1_' + self.type + '.png'
          self.visualiseClusters("Mean-Shift Clustering1", X, ms.labels_, imgName)

           # visualise 2
          imgName = img_path + 'Mean-Shift_clusters2_' + self.type + '.png'
          predict = ms.predict(X)
          self.visualiseClusters2("Mean-Shift Clustering2", X, ms.labels_, imgName, predict)

           # visualise 3
          imgName = img_path + "Mean-Shift_clusters3_" + self.type + ".png"
          self.visualiseClusters3("Mean-Shift Clustering3", X, ms.labels_, imgName, ms.cluster_centers_)
          
          return unique_labels, ms.labels_, t, "_meanshift"

     def visualiseClusters(self, title, X, labels, imgName):
          plt.scatter(X[:, 0], X[:,1], c = labels, cmap= "plasma") # plotting the clusters
          plt.title(title)
          plt.savefig(imgName)
          plt.show()

     def visualiseClusters2(self, title, X, labels, imgName, alg):
          unique_labels = set(labels)
          for i in unique_labels:
               plt.scatter(X[alg == i , 0] , X[alg == i , 1] , label = i, marker='o', picker=True)
          plt.title(title)
          plt.savefig(imgName)
          plt.show()

     def visualiseClusters3(self, title, X, labels, imgName, cluster_centers):
          no_clusters = len(np.unique(labels))
          plt.figure(1)
          plt.clf()
          colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
          for k, col in zip(range(no_clusters), colors):
               my_members = labels == k
               cluster_center = cluster_centers[k]
               plt.plot(X[my_members, 0], X[my_members, 1], col + ".")
               plt.plot(
                    cluster_center[0],
                    cluster_center[1],
                    "*",
                    markerfacecolor=col,
                    markeredgecolor="k",
                    markersize=14,
               )
          plt.title(title)
          plt.savefig(imgName)
          plt.show()
     
     def visualiseClusters4(self, title, X, labels, imgName, core_samples_mask):
          unique_labels = set(labels)
          for i in (unique_labels):
               class_member_mask = labels == i
               xy = X[class_member_mask & core_samples_mask]
               plt.plot(xy[:, 0], xy[:, 1], "o", picker=True,label = i, markeredgecolor="k")
               xy = X[class_member_mask & ~core_samples_mask]
               plt.plot(xy[:, 0], xy[:, 1], "o", picker=True, label = i,markeredgecolor="k")
          plt.title(title)
          plt.savefig(imgName)
          plt.show()
          
     def get_information(self, labels, X, unique_labels, t):
          no_clusters = len(np.unique(labels))
          no_noise = np.sum(np.array(labels) == -1, axis=0)
          print('Estimated no. of clusters: %d' % no_clusters)
          print('Estimated no. of noise points: %d' % no_noise)
          sil_score = silhouette_score(X, labels)
          db_index = davies_bouldin_score(X, labels)
          print("Silhouette Coefficient: %0.3f" % sil_score)
          print("Davies Bouldin Score: %0.3f" % db_index)

          # print("Ground Truth:", t[:,4:5]) # raw
          print("Ground Truth:", t[:,3:4]) # cloud compare
          print("Unique Labels:", unique_labels)

          write_results_to_file("*************Clustering Metrics*************")
          write_results_to_file("Silhouette Score:" + str(sil_score))
          write_results_to_file("Davies Bouldin Index:" + str(db_index))