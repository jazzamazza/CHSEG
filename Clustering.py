import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans 
from sklearn.mixture import GaussianMixture
from pyclustering.cluster import cluster_visualizer_multidim
from pyclustering.cluster.kmedians import kmedians
import pptk
from pyclustering.cluster.encoder import type_encoding, cluster_encoder
from pyclustering.cluster.center_initializer import random_center_initializer
from fcmeans import FCM       
from classification import Classification

# Clustering class with various clustering methods
class Clustering:
     def __init__(self, pointCloud, pcd_truth, pcd_choice):
          self.pcd = pointCloud
          self.pcd_truth = pcd_truth
          if (pcd_choice == "1"): self.type = "raw"
          elif (pcd_choice == "2"): self.type = "cldCmp"
          elif (pcd_choice == "3"): self.type = "pnet++"
     
     # k-means clustering 
     def k_means_clustering(self, k):
          x = self.pcd
          t = self.pcd_truth

          
          print("\n------------------k means---------------------")
          kmeans = KMeans(n_clusters=13, n_init=10) # number of clusters (k)
          kmeans.fit(x) # apply k means to dataset
          print("x[0]", x[0])
          
          # get labels 
          y_km = kmeans.predict(x)              
                    
          # get centroids 
          centroids = kmeans.cluster_centers_
          print("centroids:", centroids)
          # unique labels 
          unique_labels = np.unique(y_km)
          print("unique_labels:", unique_labels)

         #plot in matplot lib 
          for i in unique_labels:
               plt.scatter(x[y_km == i , 0] , x[y_km == i , 1] , label = i, marker='o', picker=True)
          plt.scatter(
               centroids[:, 0], centroids[:, 1],
               s=100, marker='*',
               c='red', edgecolor='black',
               label='centroids'
           )
          plt.legend()
          plt.title('Two clusters of data')
          plt.savefig('k_means_clusters.png') 
          plt.show()

          #visualise in pptk 
          xyz = self.pcd[:,0:3]
          intensity1d = (self.pcd[:,3:4]).flatten()
          view = pptk.viewer(xyz, intensity1d, y_km)
     
          return unique_labels, y_km, t, "_kmeans"

     # Gaussian mixture model clustering 
     def GMM_clustering(self):
          print("\n------------------ GMM ---------------------")
          
          # original point cloud 
          x = self.pcd
          # point cloud with the ground truth label - only used when calling the classification method
          t = self.pcd_truth

          #fit GMM 
          gm = GaussianMixture(n_components=44, covariance_type='full', random_state=0, reg_covar=0.005) 
          gm.fit(x)

          # get labels 
          labels = gm.predict(x)
          print("labels", labels)
         
          # get unique labels 
          unique_labels = np.unique(labels)

          # visualise in matplot lib 
          # for i in unique_labels:
          #      plt.scatter(x[labels == i , 0] , x[labels == i , 1] , label = i, marker='o', picker=True)
          # plt.title('Gausian Mixture Model')
          # plt.savefig('GMM_clusters.png')
          # plt.show()
     
          # visualise in pptk 
          xyz = self.pcd[:,0:3]
          intensity1d = (self.pcd[:,3:4]).flatten()
          print("intensity1d", intensity1d)
          view = pptk.viewer(xyz, labels.flatten())


          return unique_labels, labels, t, "_GMM"
         



     
     def KMedians_clustering(self):  #The algorithm is less sensitive to outliers than K-Means. Medians are calculated instead of centroids.
          print("\n------------------k medians---------------------")   

          # set data to pcd 
          data = self.pcd
          print(data[0])
          # set t to the point cloud with the truths 
          t= self.pcd_truth
          
          # Create instance of K-Medians algorithm.
          initial_medians = random_center_initializer(data,67).initialize()
          
          kmedians_instance = kmedians(data, initial_medians)

          # Run cluster analysis and obtain results.
          y_km = kmedians_instance.process()
          clusters = kmedians_instance.get_clusters()
          
          #initialise k-medians
          medians = kmedians_instance.get_medians()

           
          # get labels
          type_repr = kmedians_instance.get_cluster_encoding()
          type_encoder = cluster_encoder(type_repr, clusters, data)
          type_encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING) 
          labels_ = type_encoder.get_clusters()
          # get unique labels 
          unique_labels = np.unique(labels_)

         # visualize in pptk
          xyz = self.pcd[:,0:3]
          view = pptk.viewer(xyz, type_encoder.get_clusters())
          view.capture('screenshot.png')
          print("pptk loaded")

        # visualise in matplot lib
          vis = cluster_visualizer_multidim()
          vis.append_clusters(clusters,data.tolist(),marker="o",markersize=5)
          vis.show(pair_filter=[[0,1]], max_row_size=2)
          vis.save("clarans_clustering.png")



          return unique_labels, labels_, t, "_kmedians"

     def fuzzy_cmeans_clustering(self):
          # original point cloud 
          x = self.pcd
          # point cloud with the ground truth label - only used when calling the classification method
          t = self.pcd_truth

          print("--------------fuzzy c-means -------------")

          fcm = FCM(n_clusters=28)
          #apply fuzzy c-means to the data set 
          y_km = fcm.fit(x)

          # get labels 
          fcm_centers = fcm.centers
          fcm_labels = fcm.predict(x)

          # get unique labels 
          unique_labels = np.unique(fcm_labels)

       
         # visualise in pptk 
          xyz = self.pcd[:,0:3]
          view = pptk.viewer(xyz, fcm_labels.flatten())

         # visualise in matplot lib 
          f, axes = plt.subplots(1, 2, figsize=(11,5))
          axes[0].scatter(x[:,0], x[:,1], alpha=.1)
          axes[1].scatter(x[:,0], x[:,1], c=fcm_labels, alpha=.1)
          axes[1].scatter(fcm_centers[:,0], fcm_centers[:,1], marker="+", s=500, c='w')
          plt.savefig('fuzzy')
          plt.show()

          return unique_labels, fcm_labels, t, "_kmeans"


