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
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import pandas as pd
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

# Clustering class with various clustering methods
class Clustering:
     def __init__(self, pointCloud, pcd_choice):
          self.pcd = pointCloud
          if (pcd_choice == "1"): self.type = "raw"
          elif (pcd_choice == "2"): self.type = "cldCmp"
          elif (pcd_choice == "3"): self.type = "pnet++"
     
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
      centroids = kmeans.centroids
     
     # Visualise K-Means
      
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
      plt.savefig('k_means_clusters_' + self.type + '.png') 
      plt.show()

     #  visualizer = cluster_visualizer_multidim()
     #  visualizer.append_cluster(clusters,marker='*', markersize=10)
     #  visualizer.append_cluster(x, marker='o', markersize=10)
     #  visualizer.append_cluster(centroids, marker='*', markersize=10)
     #  visualizer.show()
     #  visualizer.save("kmeans_new_viz.png")

  
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

         

     def kMediods_clustering(self, k):

          print("\n------------------k mediods---------------------")
          
          x = self.pcd
          
         

          Kmedoids = KMedoids(n_clusters= 15,
                         metric="euclidean",
                         method="alternate",
                         init="heuristic",
                         max_iter=300,
                         random_state=None) # number of clusters (k)
          
          y_km = Kmedoids.fit_predict(x) # apply k means to dataset
          
          # Visualise K-Means
          # y_km = Kmedoids.predict(x)
          centroids = Kmedoids.cluster_centers_
          unique_labels = np.unique(y_km)
          for i in unique_labels:
               plt.scatter(x[y_km == i , 0] , x[y_km == i , 1] , label = i, marker='o', picker=True)
          plt.scatter(
               centroids[:, 0], centroids[:, 1],
               s=100, marker='*',
               c='red', edgecolor='black',
               label='centroids'
          )
          #plt.legend()
          plt.title('Two clusters of data: K-mediods')
          plt.savefig('k_mediods_clusters.png') 
          plt.show()
#To overcome the problem of sensitivity to outliers, instead of taking the mean value 
# as the centroid, we can take actual data point to represent the cluster, this is what K-medoids does.

     def GMM_clustering(self):
          print("\n------------------ GMM ---------------------")

          x = self.pcd

          gm = GaussianMixture(n_components=2)
          gm.fit(x)
          np.round(gm.weights_, 2)
          np.round(gm.means_, 2)
          # gm.predict(finalPCD)

          labels = gm.predict(x)
          print("aic", gm.aic(x))
          print("bic", gm.bic(x))

          unique_labels = np.unique(labels)
          print(unique_labels)
          for i in unique_labels:
               plt.scatter(x[labels == i , 0] , x[labels == i , 1] , label = i, marker='o', picker=True)
          
          plt.legend()
          plt.title('Gausian Mixture Model')
          plt.savefig('GMM.png')
          plt.show()

          # using AIC and BIC scores
          n_components = np.arange(1, 100)
          print("n comp", n_components)
          models = [GaussianMixture(n, 
                          covariance_type='full', random_state=0).fit(x) for n in n_components]
          models[0:5]

          # for i in n_components:
          #      GaussianMixture(n_components = i, random_state=0)
          [GaussianMixture(random_state=0),
           GaussianMixture(n_components=2, random_state=0),
           GaussianMixture(n_components=3, random_state=0),
           GaussianMixture(n_components=4, random_state=0),
           GaussianMixture(n_components=5, random_state=0)]

          gmm_model_comparisons=pd.DataFrame({"n_components" : n_components,
                                  "BIC" : [m.bic(x) for m in models],
                                   "AIC" : [m.aic(x) for m in models]})

          gmm_model_comparisons.head()

          plt.figure(figsize=(8,6))
          plt.plot(data=gmm_model_comparisons[["BIC","AIC"]])
          plt.xlabel("Number of Clusters")
          plt.ylabel("Score")
          plt.savefig("GMM_model_comparison_with_AIC_BIC_Scores_Python.png",
                    format='png',dpi=150) 
                    

# (Clustering Large Applications based upon RANdomized Search) : 
# It presents a trade-off between the cost and the effectiveness of using samples to obtain clustering.
# First, it randomly selects k objects in the data set as the current medoids. It then randomly selects a current medoid x and an object y that is not one of the current medoids.
# Then it checks for the following condition:
# Can replacing x by y improve the absolute-error criterion?
# If yes, the replacement is made. CLARANS conducts such a randomized search l times. The set of the current medoids after the l steps is considered a local optimum.
# CLARANS repeats this randomized process m times and returns the best local optimal as the final result.
     def Clarans_clustering(self):
          print("\n------------------ CLARANS ---------------------")
          data = self.pcd
          
          data = data.tolist()
          print("A peek into the dataset : ",data[:4])      # datapoints have four features 

          #@param[in] numlocal (uint): The number of local minima obtained (amount of iterations for solving the problem).
          #@param[in] maxneighbor (uint): The maximum number of neighbors examined.
          
          clarans_instance = clarans(data, 3, 4, 6);        #objects to be clustered = 3, number of obtained local minima = 6
                                                            #max number of neighboring data points examined = 4

          #calls the clarans method 'process' to implement the algortihm
          (ticks, result) = timedcall(clarans_instance.process);
          print("Execution time : ", ticks, "\n");

          #returns the clusters 
          clusters = clarans_instance.get_clusters();

          #returns the mediods 
          medoids = clarans_instance.get_medoids();
          
          print("encoding", clarans_instance.get_cluster_encoding())


          print("Index of the points that are in a cluster : ",clusters)
          print("index of the best mediods :", medoids)
          #print("The target class of each datapoint : ",data.labels)

          score = silhouette(data, clusters).process().get_score()
          print("s", score)
          max_elem  = max(score)
          print("max", max_elem)

          index = score.index(max_elem)
          print("index", index)

          # fig,axs=plt.subplots(1,2,figsize=(10,5))
          # axs[0].scatter(data[:,2],data[:,1])
          # axs[0].set_title('Synthetic clusters')
          # axs[1].set_title('CLARANS clusters')
          # for c_,m in zip(clusters,medoids):
          #      axs[1].scatter(data[c_,2],data[c_,1])
          #      axs[1].scatter(data[m,2],data[m,1],marker='x',c='black',label='medoids')
          # fig.show()
          vis = cluster_visualizer_multidim()
          vis.append_clusters(clusters,data,marker="o",markersize=5)
          vis.append_cluster(medoids,data,marker="*",markersize=5)
          vis.show(pair_filter=[[0,1]], max_row_size=2)
          vis.save("clarans_clustering.png")
          vis.show(pair_filter=[[1,2],[1,3],[27,28],[27,29]],max_row_size=2)

     
     def affinity_progpogation_clustering(self):
          x = self.pcd
          clustering = AffinityPropagation(random_state=None).fit(x)
          labels = clustering.labels_
          print("labels", labels)
          x  = x.copy(order='C')
          y_km = clustering.predict(x)

          centroids = clustering.cluster_centers_
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
          S = metrics.silhouette_score(x, labels, metric='euclidean', sample_size=None, random_state=None)
          print("s", S)
          plt.figure(figsize=(16,8), dpi=300)
          plt.plot(S, 'bo-', color='black')
          plt.xlabel('k')
          plt.ylabel('Silhouette Score')
          plt.title('Identify the number of clusters using Silhouette Score')
          plt.show()

          plt.title('clusters of data')
          plt.savefig('affinity_propogation.png') 
          plt.show()


     
     def KMedians_clustering(self):  #The algorithm is less sensitive to outliers than K-Means. Medians are calculated instead of centroids.
          print("\n------------------k medians---------------------")   

          data = self.pcd
          
          # Create instance of K-Medians algorithm.
          initial_medians = [[0.0, 0.1,0.2,0.3], [2.5, 0.7,0.8,0.9]] #@param[in] initial_medians (list): Initial coordinates of medians of clusters that are represented by list: [center1, center2, ...].
          kmedians_instance = kmedians(data, initial_medians)

          # Run cluster analysis and obtain results.
          y_km = kmedians_instance.process()
          clusters = kmedians_instance.get_clusters()
          medians = kmedians_instance.get_medians()

          print("clusters 0:", clusters[0])
          print("clusters 1:", clusters[1])
          print("data", data[0])
          print("initial medians", initial_medians[0])
          print("medians", medians[0])

          unique_labels = np.unique(y_km)
          print("unique_labels:", unique_labels)
          for i in unique_labels:
               plt.scatter(data[y_km == i , 0] , data[y_km == i , 1] , label = i, marker='o', picker=True)
          plt.scatter(
               medians[:][0], medians[:][1],
               s=100, marker='*',
               c='red', edgecolor='black',
               label='centroids'
          )
          #plt.legend()
          plt.title('Two clusters of data')
          plt.savefig('k_means_clusters.png') 
          plt.show()

          #commenting out everything and just appending cluster gives graph but not right 
          #also works if you comment out clusters and leave the rest
          # visualizer = cluster_visualizer_multidim()
          # # visualizer.append_cluster(clusters,marker='*', markersize=10)
          # visualizer.append_cluster(data,marker='*', markersize=10)
          # visualizer.append_cluster(initial_medians, marker='*', markersize=10)
          # visualizer.append_cluster(medians, marker='*', markersize=10)
          # visualizer.show()

          # Visualize clustering results.
          # vis = cluster_visualizer_multidim()
          # vis.append_clusters(clusters,data,marker="*",markersize=5)
          # vis.append_cluster(initial_medians, marker='*', markersize=5)
          # vis.append_cluster(medians, marker='*', markersize=5)
          # vis.show()
#https://github.com/annoviko/pyclustering/blob/master/pyclustering/cluster/kmedians.py


     def silhouette(self):
    
          x = self.pcd
     
          K = range(2, 80)
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
               clusterer = GaussianMixture(n_clusters= k, random_state=10)      #for k-means and k-medoids
               
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

     # Here is the Silhouette analysis done on the above plots to select an optimal value for n_clusters.
     # The value of n_clusters as 4 and 5 looks to be suboptimal for the given data due to the following reasons:
     # Presence of clusters with below-average silhouette scores 
     # Wide fluctuations in the size of the silhouette plots.
     # The value of 2 and 3 for n_clusters looks to be the optimal one. 
     # The silhouette score for each cluster is above average silhouette scores. Also, the fluctuation in size is similar. 
     # The thickness of the silhouette plot representing each cluster also is a deciding point. For the plot with n_cluster 3 (top right), 
     # the thickness is more uniform than the plot with n_cluster as 2 (top left) with one cluster thickness much more than the other. Thus, one can select the optimal number of clusters as 3.

     def silhouete_GMM(self):
          S=[]

          # Range of clusters to try (2 to 10)
          K=range(2,8)

          # Select data for clustering model
          X = self.pcd

          for k in K:
          # Set the model and its parameters
               model = GaussianMixture(n_components=k, n_init=20, init_params='kmeans')
                # Fit the model 
               labels = model.fit_predict(X)
               # Calculate Silhoutte Score and append to a list
               print("labels", labels)
               S.append(metrics.silhouette_score(X, labels, metric='euclidean'))

          # Plot the resulting Silhouette scores on a graph
          plt.figure(figsize=(16,8), dpi=300)
          plt.plot(K, S, 'bo-', color='black')
          plt.xlabel('k')
          plt.ylabel('Silhouette Score')
          plt.title('Identify the number of clusters using Silhouette Score')
          plt.show()