from PointCloudLoader import PointCloudLoader
from sklearn.cluster import DBSCAN, OPTICS, cluster_optics_dbscan
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt 
from sklearn.metrics import *
import numpy as np
from kneed import KneeLocator
from itertools import cycle
from sklearn.cluster import MeanShift, estimate_bandwidth

class LeahClustering:
     def __init__(self, pointCloud="", pcd_choice="1"):
          self.pcd = pointCloud
          self.type = ""
          if (pcd_choice == "1"): self.type = "raw"
          elif (pcd_choice == "2"): self.type = "cldCmp"
          elif (pcd_choice == "3"): self.type = "pnet++"
     
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
          # Run the fit
          #clust = clust.fit(X)
          #print('labels:', labels)
          space = np.arange(len(X))
          reachability = clust.reachability_[clust.ordering_]
          labels = clust.labels_#[clust.ordering_]
          print('labels2:', labels)

          self.get_information(labels, X)

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
          plt.savefig("OPTICS_reachability.png")
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
          plt.savefig("Optics_clusters1_" + self.type + '.png')
          plt.show()

          # visualise
          imgName = 'Optics_clusters_' + self.type + '.png'
          self.visualiseClusters("Optics Clustering", X, labels, imgName)
          
     # DBSCAN 
     def dbscan_clustering(self):
          print("***************DBSCAN CLUSTERING***************")
          print("starting dbscan_clustering")
          X = self.pcd
          e = self.calculateElbow()
          print("e=",e)
          min_samples_ = 30 # for cloud compare with 15 features
          min_samples_ = 4 # for raw point cloud
          db1 = DBSCAN(eps=e, min_samples=min_samples_)
          db = db1.fit(X)
          predict = db.fit_predict(X)
          print("labels:", db.labels_)
          
          core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
          core_samples_mask[db.core_sample_indices_] = True

          self.get_information(db.labels_, X)

          # visualise
          imgName = 'DBSCAN_clusters_' + self.type + '.png'
          self.visualiseClusters("DBSCAN-Shift Clustering", X, db.labels_, imgName)
          
          # visualise 2
          imgName = 'DBSCAN_clusters2_' + self.type + '.png'
          self.visualiseClusters2(self, "DBSCAN-Shift Clustering2", X, db.labels_, imgName, predict)
          
           # visualise 4
          imgName = "DBSCAN_clusters1_" + self.type + '.png'
          self.visualiseClusters4("DBSCAN Clustering 4", X, db.labels_, imgName. core_samples_mask)
          
          print("finished dbscan_clustering")

     def calculateElbow(self):
           # FIND OPTIMAL EPSILON VALUE: use elbow point detection method 
          df = self.pcd
          nearest_neighbors = NearestNeighbors(n_neighbors=5)
          neighbors = nearest_neighbors.fit(df)
          distances, indices = neighbors.kneighbors(df)
          distances = np.sort(distances[:,4], axis=0)
          fig = plt.figure(figsize=(5, 5))
          plt.plot(distances)
          plt.xlabel("Points")
          plt.ylabel("Distance")
          plt.savefig('DBSCAN_Eps.png')
          plt.show()
          
          # IDENTIFY ELBOW POINT:
          print("Identify Elbow Point:")
          i = np.arange(len(distances))
          knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
          knee.plot_knee()
          plt.xlabel("Points")
          plt.ylabel("Distance")
          plt.show()
          plt.savefig('DBSCAN_elbow.png')
          eps = distances[knee.knee]
          return eps

     def mean_shift_clustering(self):
          print("***************MEAN-SHIFT CLUSTERING***************")
          X = self.pcd
          bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
          ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
          ms.fit(X)

          self.get_information(ms.labels_, X)

          # visualise 1
          imgName = 'Mean-Shift_clusters1_' + self.type + '.png'
          self.visualiseClusters("Mean-Shift Clustering1", X, ms.labels_, imgName)

           # visualise 2
          imgName = 'Mean-Shift_clusters2_' + self.type + '.png'
          predict = ms.predict(X)
          self.visualiseClusters2("Mean-Shift Clustering2", X, ms.labels_, imgName, predict)

           # visualise 3
          imgName = "Mean-Shift_clusters3_" + self.type + ".png"
          self.visualiseClusters3("Mean-Shift Clustering3", X, ms.labels_, imgName, ms.cluster_centers_)
          
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
          
     def get_information(self, labels, X):
          no_clusters = len(np.unique(labels))
          no_noise = np.sum(np.array(labels) == -1, axis=0)
          print('Estimated no. of clusters: %d' % no_clusters)
          print('Estimated no. of noise points: %d' % no_noise)
          print("Silhouette Coefficient: %0.3f" % silhouette_score(X, labels))

if __name__ == '__main__':
    file_path = "/content/drive/Shareddrives/Thesis/Data/church_registered.npy"
    #file_path = "/content/drive/Shareddrives/Thesis/Data/church_registered_cloudCompare_100k.las" 
    pc_loader = PointCloudLoader(file_path)

    pointCloud = pc_loader.load_point_cloud_npy(False)
    # pointCloud = pc_loader.load_point_cloud_las(False)
    #pointCloud = ""

    c = LeahClustering(pointCloud)
    #c.optics_clustering()
    c.dbscan_clustering()
    #c.mean_shift_clustering()
