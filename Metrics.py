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
from sklearn_extra.cluster import KMedoids #pip install https://github.com/scikit-learn-contrib/scikit-learn-extra/archive/master.zip
#from sklearn_extra.cluster import KMedians
import sklearn_extensions as ske
from sklearn.metrics.pairwise import (
    pairwise_distances,
    pairwise_distances_argmin,
)
from sklearn.mixture import GaussianMixture
from pyclustering.cluster.clarans import clarans;
from pyclustering.utils import timedcall;
from pyclustering.cluster import cluster_visualizer_multidim
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from pyclustering.cluster.silhouette import silhouette
from sklearn.metrics import davies_bouldin_score

class Testing:
     def __init__(self, pointCloud, pcd_choice):
          self.pcd = pointCloud
        #   if (pcd_choice == "1"): self.type = "raw"
        #   elif (pcd_choice == "2"): self.type = "cldCmp"
        #   elif (pcd_choice == "3"): self.type = "pnet++"

     def silhouette_kmeans(self):
    
          x = self.pcd
     
          K = range(2, 20)
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
               clusterer = KMeans(n_clusters= k)      #for k-means and k-medoids
               
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

     def db_index(self):
          x = self.pcd

          results = {}
          
          for i in range(2,100):
               kmeans = KMeans(n_clusters=i, random_state=30)
               labels = kmeans.fit_predict(x)
               db_index = davies_bouldin_score(x, labels)
               results.update({i: db_index})
               print({i: db_index})

          plt.plot(list(results.keys()), list(results.values()))
          plt.xlabel("Number of clusters")
          plt.ylabel("Davies-Boulding Index")
          plt.show()
     