from curses import raw
from operator import truth
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, dbscan
import open3d as o3d
from mpl_toolkits import mplot3d
from datetime import datetime
from yaml import load
import laspy as lp
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN
import faiss

# Clustering class with various clustering methods
class Clustering:
     def __init__(self, pointCloud):
          self.pcd = pointCloud
     
     # K-MEANS CLUSTERING USING FAISS LIBRARY - SPEEDS UP COMPUTATION
     def k_means_clustering_faiss(self, k):
      x = self.pcd
      print("starting faiss_k_means")
      # train:
      #k = 15
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
      unique_labels = np.unique(y_km)
      for i in unique_labels:
          plt.scatter(x[y_km == i , 0] , x[y_km == i , 1] , label = i, marker='o', picker=True)
      plt.scatter(
          centroids[:, 0], centroids[:, 1],
          s=100, marker='*',
          c='red', edgecolor='black',
          label='centroids'
      )
      plt.title('K-Means Clustering')
      plt.savefig('k_means_clusters_las.png') 
      plt.show()
  
     # k means clustering method --> clusters a dataset into k (given) clusters
     def k_means_clustering(self, k):
          x = self.pcd
          
          print("\n------------------k means---------------------")
          kmeans = KMeans(n_clusters=k, n_init=10) # number of clusters (k)
          kmeans.fit(x) # apply k means to dataset
          
          # Visualise K-Means
          y_km = kmeans.predict(x)
          print("y_km:", y_km)
          centroids = kmeans.cluster_centers_
          print("centroids:", centroids)
          unique_labels = np.unique(y_km)
          print("unique_labels:", unique_labels)
          for i in unique_labels:
               plt.scatter(x[y_km == i , 0] , x[y_km == i , 1] , label = i, marker='o', picker=True)
          plt.scatter(
               centroids[:, 0], centroids[:, 1],
               s=100, marker='*',
               c='red', edgecolor='black',
               label='centroids'
          )
          #plt.legend()
          plt.title('Two clusters of data')
          plt.savefig('k_means_clusters.png') 
          plt.show()

     def silhouette(self):
    
          x = self.pcd
     
          K = range(2,10)
          for k in K:
               fig, (ax1, ax2) = plt.subplots(1, 2)
               fig.set_size_inches(18, 7)

               # The 1st subplot is the silhouette plot
               # The silhouette coefficient can range from -1, 1 but in this example all
               # lie within [-0.1, 1]
               ax1.set_xlim([-0.1, 1])
               # The (n_clusters+1)*10 is for inserting blank space between silhouette
               # plots of individual clusters, to demarcate them clearly.
               ax1.set_ylim([0, len(x) + (k + 1) * 10])

               # Initialize the clusterer with n_clusters value and a random generator
               # seed of 10 for reproducibility.
               clusterer = KMeans(n_clusters= k, random_state=10)
               cluster_labels = clusterer.fit_predict(x)

               silhouette_avg = silhouette_score(x, cluster_labels)
               print(
                    "For n_clusters =",
                         k,
                    "The average silhouette_score is :",
                         silhouette_avg,
               )
               sample_silhouette_values = silhouette_samples(x, cluster_labels)
     
               y_lower = 10
               for i in range(k):
                    # Aggregate the silhouette scores for samples belonging to
                    # cluster i, and sort them
                    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                    ith_cluster_silhouette_values.sort()

                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i

                    color = cm.nipy_spectral(float(i) / k)
                    ax1.fill_betweenx(
                         np.arange(y_lower, y_upper),
                         0,
                         ith_cluster_silhouette_values,
                         facecolor=color,
                         edgecolor=color,
                         alpha=0.7,
                    )

                    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                    # Compute the new y_lower for next plot
                    y_lower = y_upper + 10  # 10 for the 0 samples

               ax1.set_title("The silhouette plot for the various clusters.")
               ax1.set_xlabel("The silhouette coefficient values")
               ax1.set_ylabel("Cluster label")

               ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

               ax1.set_yticks([])  # Clear the yaxis labels / ticks
               ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

               # 2nd Plot showing the actual clusters formed
               colors = cm.nipy_spectral(cluster_labels.astype(float) / k)
               ax2.scatter(
                    x[:, 0], x[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
               )

               # Labeling the clusters
               centers = clusterer.cluster_centers_
               # Draw white circles at cluster centers
               ax2.scatter(
                    centers[:, 0],
                    centers[:, 1],
                    marker="o",
                    c="white",
                    alpha=1,
                    s=200,
                    edgecolor="k",
               )

               for i, c in enumerate(centers):
                    ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

               ax2.set_title("The visualization of the clustered data.")
               ax2.set_xlabel("Feature space for the 1st feature")
               ax2.set_ylabel("Feature space for the 2nd feature")

               plt.suptitle(
                    "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
                    % k,
                    fontsize=14,
                    fontweight="bold",
               )

          plt.show()

     #output
     # For n_clusters = 2 The average silhouette_score is : 0.3889045527426348
     # For n_clusters = 3 The average silhouette_score is : 0.3403178007585951
     # For n_clusters = 4 The average silhouette_score is : 0.31642498946572917
     # For n_clusters = 5 The average silhouette_score is : 0.30732018048203114
     # For n_clusters = 6 The average silhouette_score is : 0.32511128283087165
     # For n_clusters = 7 The average silhouette_score is : 0.326443764304626
     # For n_clusters = 8 The average silhouette_score is : 0.329213106052044
     # For n_clusters = 9 The average silhouette_score is : 0.31500958433214615