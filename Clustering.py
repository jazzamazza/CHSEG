from enum import unique
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans 
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
from sklearn.metrics import davies_bouldin_score
from scipy.spatial.distance import pdist
from itertools import cycle
import pptk
# from sklearnex import patch_sklearn
# patch_sklearn()
import statistics
# from KMediansPy.distance import distance
# from KMediansPy.KMedians import KMedians
# from KMediansPy.summary import summary
from pyclustering.cluster.encoder import type_encoding, cluster_encoder
from pyclustering.cluster.silhouette import silhouette_ksearch_type, silhouette_ksearch
from pyclustering.cluster.center_initializer import random_center_initializer
from pyclustering.cluster.fcm import fcm
from fcmeans import FCM       #pip install fuzzy-c-means
#from pyclustering.cluster.kmedians import get_medians
# from daal4py.oneapi import sycl_context
import time
from datetime import datetime
# Clustering class with various clustering methods
class Clustering:
     def __init__(self, pointCloud, pcd_truth, pcd_choice):
          self.pcd = pointCloud
          self.pcd_truth = pcd_truth
          if (pcd_choice == "1"): self.type = "raw"
          elif (pcd_choice == "2"): self.type = "cldCmp"
          elif (pcd_choice == "3"): self.type = "pnet++"
     
     def get_ground_truth(self, unique_labels, y_km, t):
      
          num_keep, num_discard = 0, 0
    
          ground_truths = np.array([])
          print("t[0]", t[0])
          print("ground_truth size:", ground_truths.size)
          for i in unique_labels:
              num_keep, num_discard = 0, 0
              #print("cluster:", i)
              #for point, p in map(None, x[y_km == i], t[y_km == i]):
              for point in t[y_km == i]:
                #print("p", point[4])
                if (point[4] >= float(0.5)): num_discard += 1
                else: num_keep += 1
              print("num_keep:", num_keep)
              print("num_discard:", num_discard)
              if num_keep > num_discard: 
                ground_truths = np.append(ground_truths, 1)      #changing the clusters to keep and discard
              else: 
                ground_truths = np.append(ground_truths, 0)

          print("ground_truth:", ground_truths)
          
          #accounting for the points that arent 1 or 0 and are in between 
          g = np.asarray(t)        
          for i in range(0, len(ground_truths)):   #i is each cluster
            if ground_truths[i] == float(1): # if cluster == keep
              for point in t[y_km == i]: # set ground truth of each point to keep
                t[y_km == i, 4:5] = float(1)
            else:
              for point in t[y_km == i]:
                t[y_km == i, 4:5] = float(0)
                
          print("t shape", np.shape(t))
          print("truth", t[0])
          print("g", g[0])
          # print("t", t)

          # for i in unique_labels:
          #     print("cluster:", i)
          #     for point in t[y_km == i]:
          #       print("new point", t[y_km == i, 4:5])

          #np.save('/content/drive/Shareddrives/CHSEG/data/gmm_t_0.5', t)
          print("t[:,4:5].flatten()", t[:,4:5].flatten())
          xyz = self.pcd[:,0:3]
          intensity1d = (self.pcd[:,3:4]).flatten()
          view = pptk.viewer(xyz, intensity1d, t[:,4:5].flatten())  #t[:,5:6].flatten()
          print("pptk loaded")

     def get_ground_truth_cloud_comp(self, unique_labels, y_km, t):
      
          num_keep, num_discard = 0, 0
    
          ground_truths = np.array([])
          print("ground_truth size:", ground_truths.size)
          for i in unique_labels:
              num_keep, num_discard = 0, 0
              #print("cluster:", i)
              # for point, p in map(None, x[y_km == i], t[y_km == i]):
              for point in t[y_km == i]:
                #print("p", point[4])
                if (point[15] >= float(0.5)): num_discard += 1
                else: num_keep += 1
              print("num_keep:", num_keep)
              print("num_discard:", num_discard)
              if num_keep > num_discard: 
                ground_truths = np.append(ground_truths, 1)
              else: 
                ground_truths = np.append(ground_truths, 0)

          print("ground_truth:", ground_truths)
          
     
          g = np.asarray(t)
          for i in range(0, len(ground_truths)):   #i is each cluster
            if ground_truths[i] == float(1): # if cluster == keep
              for point in t[y_km == i]: # set ground truth of each point to keep
                t[y_km == i, 15:16] = float(1)
            else:
              for point in t[y_km == i]:
                t[y_km == i, 15:16] = float(0)
          print("t shape", np.shape(t))
          print("t[0]", t[0])

          # for i in unique_labels:
          #     print("cluster:", i)
          #     for point in t[y_km == i]:
          #       print("new point", t[y_km == i, 4:5])

          #np.save('/content/drive/Shareddrives/CHSEG/data/gmm_new_t_0.5', t)

          xyz = self.pcd[:,0:3]
          intensity1d = (self.pcd[:,15:16]).flatten()
          view = pptk.viewer(xyz, intensity1d, t[:,15:16].flatten())  #t[:,5:6].flatten()
          print("pptk loaded")

     
     # K-MEANS CLUSTERING USING FAISS LIBRARY - SPEEDS UP COMPUTATION
     def k_means_clustering_faiss(self, k):
      x = self.pcd
      x  = x.copy(order='C')
      print("starting faiss_k_means")
      print("x[0]", x[0])
      print("x[1]", x[1])
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


     #  db_index = davies_bouldin_score(x, y_km)
     #  print(db_index)
     
     # Visualise K-Means
      xyz = self.pcd[:,0:3]
      intensity1d = (self.pcd[:,3:4]).flatten()
      view = pptk.viewer(xyz, intensity1d, y_km)
      print("pptk loaded")
     
      zero = x[y_km == 0 , :]       #printing out x,y,z of
     #  one = x[y_km == 1 , :] , x[y_km == 1 , :]
     #  two = x[y_km == 2 , :] , x[y_km == 2 , :])

      print("zero", zero)
    
      unique_labels = np.unique(y_km)
      final_y_km = []
      for i in unique_labels:
          final_y_km.append(x[y_km == i , :]) #: is the rest of it , for every point printing ut x,y,z intesnisty
          print("final", final_y_km[0][:])
          print("in loop:", x[y_km == i , 0] , x[y_km == i , 1])
          plt.scatter(x[y_km == i , 0] , x[y_km == i , 1] , label = i, marker='o', picker=True)
      plt.scatter(
          centroids[:, 0], centroids[:, 1],
          s=100, marker='*',
          c='red', edgecolor='black',
          label='centroids'
      )
      
      print("x and final_y_km EQUAL?",np.array_equal(x, final_y_km))
      plt.title('K-Means Clustering')
      plt.savefig('k_means_clusters_' + self.type + '.png') 
      plt.show()


  
     # k means clustering method --> clusters a dataset into k (given) clusters

     def k_means_clustering(self, k):
          x = self.pcd
          t = self.pcd_truth

          
          print("\n------------------k means---------------------")
          kmeans = KMeans(n_clusters=k, n_init=10) # number of clusters (k)
          kmeans.fit(x) # apply k means to dataset
          print("x[0]", x[0])
          # Visualise K-Means
          y_km = kmeans.predict(x)
          print("y_km:", y_km)               #10 clusters
                    
          
          centroids = kmeans.cluster_centers_
          print("centroids:", centroids)
          unique_labels = np.unique(y_km)
          print("unique_labels:", unique_labels)

          #get ground truth 
          print("t in kmeans", t[0])
          Clustering.get_ground_truth(self, unique_labels, y_km, t)
          print("get ground truth")

          for i in unique_labels:
               #print((x[y_km == i , 0] , x[y_km == i , 1])) #how to access the pointd 
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

          Kmedoids = KMedoids(n_clusters= 14,
                         metric="euclidean",
                         method="alternate",
                         init="k-medoids++",
                         max_iter=300,
                         random_state=0) # number of clusters (k)
          
          y_km = Kmedoids.fit_predict(x) # apply k means to dataset
          print("y_km", y_km[0])
          print("x", x[0])

          mediods1d = y_km

          #print("shape", mediods1d.shape())
          xyz = x[:,:3]
          intensity1d = (x[:,3:4]).flatten()
          print("shape", np.shape(mediods1d))
          view = pptk.viewer(xyz, intensity1d, mediods1d)
          print("pptk loaded")
          

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
          plt.title('2 clusters of data: K-mediods')
          plt.savefig('k_mediods_clusters.png') 
          plt.show()
#To overcome the problem of sensitivity to outliers, instead of taking the mean value 
# as the centroid, we can take actual data point to represent the cluster, this is what K-medoids does.

# try visualize
# then try change covariance to diag and spherical - important estimator
     def GMM_clustering(self):
          print("\n------------------ GMM ---------------------")

          x = self.pcd
          truth = self.pcd_truth

          now = datetime.now()
          current_time = now.strftime("%H:%M:%S")
          print("Current Time =", current_time)
         
          

          print("x[0]", x[0])
          print("truth", truth[0])

          gm = GaussianMixture(n_components=46, covariance_type='spherical', random_state=0) #parameters dont make a dif to sil and db index
          gm.fit(x)
          # np.round(gm.weights_, 2)
          # np.round(gm.means_, 2)

          labels = gm.predict(x)
          print("labels", labels)

          print("==================================")
          print("done clustering")

          now = datetime.now()
          current_time = now.strftime("%H:%M:%S")
          print("Current Time =", current_time)
         

         

          unique_labels = np.unique(labels)


          # print(unique_labels)
          # for i in unique_labels:
          #      plt.scatter(x[labels == i , 0] , x[labels == i , 1] , label = i, marker='o', picker=True)
          # plt.title('Gausian Mixture Model')
          # plt.savefig('GMM_clusters.png')
          # plt.show()
          
          print("==================================")
          print("done vis")
          

          # sil = silhouette_score(x, labels, metric='euclidean')
          # print("sil", sil)

          # db  = davies_bouldin_score(x, labels)
          # print("db", db)

          print("==================================")
          print("done scores")

          print('TRUTH IN GMM', truth[0])
          print("unique labels", unique_labels)

          # Clustering.get_ground_truth(self, unique_labels, labels, truth)
          # print("getting ground truth")

          # probs = gm.predict_proba(x)
          # print(probs[:5].round(3))
          # size = 50 * probs.max(1) ** 2  # square emphasizes differences
          # plt.scatter(x[:, 0], x[:, 1], c=labels, cmap='viridis', s=size);
          # plt.show()
         
          
          # xyz = self.pcd[:,0:3]
          # intensity1d = (self.pcd[:,3:4]).flatten()
          # view = pptk.viewer(xyz, intensity1d, labels)
          # print("pptk loaded")
         

         

          # using AIC and BIC scores
          print("aic", gm.aic(x))
          print("bic", gm.bic(x))

          n_components = np.arange(1, 50)
          print("n comp", n_components)
          models = [GaussianMixture(n, 
                          covariance_type='full', random_state=0).fit(x) for n in n_components]

          plt.plot(n_components, [m.bic(x) for m in models], label='BIC')
          plt.legend(loc='best')
          plt.xlabel('n_components')
          plt.show()

          gradient = np.gradient([m.bic(x) for m in models])
          plt.plot(n_components, gradient)
          plt.legend(loc='best')
          plt.xlabel('n_components')
          plt.show()
         
          # models[0:5]

          # # for i in n_components:
          # #      GaussianMixture(n_components = i, random_state=0)
          # [GaussianMixture(random_state=0),
          #  GaussianMixture(n_components=2, random_state=0),
          #  GaussianMixture(n_components=3, random_state=0),
          #  GaussianMixture(n_components=4, random_state=0),
          #  GaussianMixture(n_components=5, random_state=0)]

          # gmm_model_comparisons=pd.DataFrame({"n_components" : n_components,
          #                         "BIC" : [m.bic(x) for m in models],
          #                          "AIC" : [m.aic(x) for m in models]})

          # gmm_model_comparisons.head()

          # plt.figure(figsize=(8,6))
          # plt.plot(data=gmm_model_comparisons[["BIC","AIC"]])
          # plt.xlabel("Number of Clusters")
          # plt.ylabel("Score")
          # plt.savefig("GMM_model_comparison_with_AIC_BIC_Scores_Python.png",
          #           format='png',dpi=150) 
                    

# (Clustering Large Applications based upon RANdomized Search) : 
# It presents a trade-off between the cost and the effectiveness of using samples to obtain clustering.
# First, it randomly selects k objects in the data set as the current medoids. It then randomly selects a current medoid x and an object y that is not one of the current medoids.
# Then it checks for the following condition:
# Can replacing x by y improve the absolute-error criterion?
# If yes, the replacement is made. CLARANS conducts such a randomized search l times. The set of the current medoids after the l steps is considered a local optimum.
# CLARANS repeats this randomized process m times and returns the best local optimal as the final result.
     def Clarans_clustering(self):
          print("\n------------------ CLARANS ---------------------")
          now = datetime.now()
          current_time = now.strftime("%H:%M:%S")
          print("Current Time =", current_time)


          data = self.pcd
          
          data = data.tolist()
          print("A peek into the dataset : ",data[:4])      # datapoints have four features 

          #@param[in] numlocal (uint) 3rd param: The number of local minima obtained (amount of iterations for solving the problem).
          #@param[in] maxneighbor (uint) 4th param: The maximum number of neighbors examined and no. of clusters to be formed ( k ) as input.
          
          clarans_instance = clarans(data, 3, 1, 1);        #objects to be clustered = 3, number of obtained local minima = 6
          #reduce max neighbour 
          #make num local 1 - increasing it to 2 makes time increase exponentially                                             #max number of neighboring data points examined = 4
                                                            

          #calls the clarans method 'process' to implement the algortihm
          (ticks, result) = timedcall(clarans_instance.process);
          print("Execution time : ", ticks, "\n");

          #returns the clusters 
          clusters = clarans_instance.get_clusters();

          #returns the mediods 
          medoids = clarans_instance.get_medoids();
          
          # print("encoding", clarans_instance.get_cluster_encoding())
          # print("Index of the points that are in a cluster : ",clusters)
          # print("index of the best mediods :", medoids)
          #print("The target class of each datapoint : ",data.labels)

          type_repr = clarans_instance.get_cluster_encoding()
          type_encoder = cluster_encoder(type_repr, clusters, data)
          type_encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
          print("Index Labeling:", type_encoder.get_clusters())                 #gets labels of clusters

          score = silhouette(data, clusters).process().get_score()
          print("s", score)
          # max_elem  = max(score)
          # print("max", max_elem)

          # index = score.index(max_elem)
          # print("index", index)
     
          print("==================================")
          now = datetime.now()
          current_time = now.strftime("%H:%M:%S")
          print("Current Time clustering =", current_time)

          

          # vis = cluster_visualizer_multidim()
          # vis.append_clusters(clusters,data,marker="o",markersize=5)
          # vis.append_cluster(medoids,data,marker="*",markersize=5)
          # vis.show(pair_filter=[[0,1]], max_row_size=2)
          # vis.save("clarans_pic.png")

          print("==================================")
          print("done vis")

          now = datetime.now()
          current_time = now.strftime("%H:%M:%S")
          print("Current Time starting SC to 50 =", current_time)

          search_instance = silhouette_ksearch(data, 2, 10, algorithm=silhouette_ksearch_type.KMEDOIDS).process()

          amount = search_instance.get_amount()
          scores = search_instance.get_scores()
          print("amount", amount)
          print("Scores: '%s'" % str(scores))

          print("==================================")
          print("done scores")

          now = datetime.now()
          current_time = now.strftime("%H:%M:%S")
          print("Current Time ending SC to 50 =", current_time)


          # xyz = self.pcd[:,0:3]
          # intensity1d = (self.pcd[:,3:4]).flatten()
          # view = pptk.viewer(xyz, intensity1d, type_encoder.get_clusters())
          # view.capture('screenshot.png')
          # print("pptk loaded")
          #vis.show(pair_filter=[[1,2],[1,3],[27,28],[27,29]],max_row_size=2)


     #it is advised to start with a preference equal to the median of your data
     


     
     def KMedians_clustering(self):  #The algorithm is less sensitive to outliers than K-Means. Medians are calculated instead of centroids.
          print("\n------------------k medians---------------------")   

          now = datetime.now()
          current_time = now.strftime("%H:%M:%S")
          print("Current Time =", current_time)

          data = self.pcd
          print(data[0])
          truth = self.pcd_truth
          
          # Create instance of K-Medians algorithm.
          # initial_medians = [[0.0, 0.1,0.2,0.3], [2.5, 0.7,0.8,0.9]] #@param[in] initial_medians (list): Initial coordinates of medians of clusters that are represented by list: [center1, center2, ...].
          #initial_medians = [[0,  0,  0 , 0]]
          initial_medians = random_center_initializer(data, 2).initialize()
          
          # initial_medians = [[ -2.93801735,   8.83804056,   1.8017709,    0.64767548],
          #                    [ -3.00039742,  -2.97420928,  -0.45340406,   0.71791216],
          #                    [-25.42512459,   6.8176853,    2.88496126,   0.79715821],
          #                    [ 0.08984507,   1.02368482,  -3.07110033,   0.71199397],
          #                    [ 9.7704546,   19.90037858,   3.350909,     0.69528395],
          #                     [ 3.539814 ,   -1.69007941,  -0.06266233,   0.73405861],
          #                     [-19.14666025,  37.00354426,   1.3612698,    0.87105943],
          #                     [ 18.59309298,  -7.48216544,  9.48487367,   0.77012153],
          #                     [ -2.91215695,   2.56525525,   0.33262204,   0.68599992],
          #                     [  2.55001398,   5.21509668,   1.13228612,   0.69533387],
          #                     [  2.73877996,  12.02266425,   2.29875192,   0.71051266],
          #                     [ 19.1447,     -53.2159,     -48.148,        1.        ], 
          #                     [ -2.67284345,  -19.32162145,   7.51331423 ,  0.87050616]] #13 medians means 13 clusters 
          kmedians_instance = kmedians(data, initial_medians)

          #y_km = kmedians.predict(data, initial_medians)

          # Run cluster analysis and obtain results.
          y_km = kmedians_instance.process()
          print("h")
          clusters = kmedians_instance.get_clusters()
          print("shape", np.shape(clusters)[0])             #how many clusters 
          
          for i in clusters:            #how many points in each cluster
               shape = np.shape(i)[0]
               print("number of points in each cluster:", shape)
          
          cluster1 = kmedians_instance.get_clusters()[0]
          print("cluster1", cluster1)
          medians = kmedians_instance.get_medians()
        

          
          print("initial medians", initial_medians[0])
          print("data", data[0])
          print("medians[0]", medians[:][0])

          print("==================================")
          print("done clustering")

          now = datetime.now()
          current_time = now.strftime("%H:%M:%S")
          print("Current Time =", current_time)
          
          type_repr = kmedians_instance.get_cluster_encoding()
          type_encoder = cluster_encoder(type_repr, clusters, data)
          type_encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
          print("Index Labeling:", type_encoder.get_clusters())
          
          labels_ = type_encoder.get_clusters()
          print("labels", labels_)
          unique_labels = np.unique(labels_)
          print("unique labels", unique_labels)
          

          Clustering.get_ground_truth(self, unique_labels, labels_, truth)
          print("getting ground truth")

          # score = silhouette(data, clusters).process().get_score()
          # #print("s", score)
          # max_elem  = max(score)
          # print("max", max_elem)

          # index = score.index(max_elem)
          # print("index", index)

        
          # encoder = cluster_encoder(type_repr, clusters, data);

          # xyz = self.pcd[:,0:3]
          # intensity1d = (self.pcd[:,3:4]).flatten()
          # view = pptk.viewer(xyz, intensity1d, type_encoder.get_clusters())
          # view.capture('screenshot.png')
          # print("pptk loaded")





          vis = cluster_visualizer_multidim()
          vis.append_clusters(clusters,data.tolist(),marker="o",markersize=5)
          # vis.append_cluster(medians,data,marker="*",markersize=5)
          vis.show(pair_filter=[[0,1]], max_row_size=2)
          vis.save("clarans_clustering.png")

          print("==================================")
          print("done vis")


          # for i in unique_labels:
          #      plt.scatter(data[y_km == i , 0] , data[y_km == i , 1] , label = i, marker='o', picker=True)
          # plt.scatter(
          #      medians[:][0], medians[:][0],
          #      s=100, marker='*',
          #      c='red', edgecolor='black',
          #      label='centroids'
          # )
          # #plt.legend()
          # plt.title('Two clusters of data')
          # plt.savefig('k_means_clusters.png') 
          # plt.show()

          #commenting out everything and just appending cluster gives graph but not right 
          #also works if you comment out clusters and leave the rest
          # visualizer = cluster_visualizer_multidim()
          # print("h")
          # visualizer.append_cluster(clusters,marker='o', markersize=10)
          # print("J")
          # visualizer.append_cluster(data,marker='o', markersize=10)
          # # visualizer.append_cluster(initial_medians, marker='*', markersize=10)
          # # visualizer.append_cluster(medians, marker='*', markersize=10)
          # print("K")
          # visualizer.show()
          # print("L")
          # visualizer.save("kmedians")

          # Visualize clustering results.
          # vis = cluster_visualizer_multidim()
          # vis.append_clusters(clusters,data,marker="*",markersize=5)
          # vis.append_cluster(initial_medians, marker='*', markersize=5)
          # vis.append_cluster(medians, marker='*', markersize=5)
          # vis.show()
#https://github.com/annoviko/pyclustering/blob/master/pyclustering/cluster/kmedians.py

    
     def tryfuzzy(self):
          
          data = self.pcd
 
          # initialize
          initial_centers = kmeans_plusplus_initializer(data, 2, kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE).initialize()
 
          # create instance of Fuzzy C-Means algorithm
          fcm_instance = fcm(data, initial_centers)
 
          # run cluster analysis and obtain results
          fcm_instance.process() #y_km <pyclustering.cluster.fcm.fcm object at 0x14ea224a8>
          clusters = fcm_instance.get_clusters()
          centers = fcm_instance.get_centers()

          type_repr = fcm_instance.get_cluster_encoding()
          type_encoder = cluster_encoder(type_repr, clusters, data)
          type_encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
          print("Index Labeling:", type_encoder.get_clusters())

          print("==================================")
          print("done clustering")

           #visualize clustering results
          visualizer = cluster_visualizer_multidim()
          visualizer.append_clusters(type_encoder.get_clusters, data,  marker='o',  markersize=5)
          visualizer.append_cluster(centers, marker='*', markersize=10)
          visualizer.show()
          visualizer.save("fuzzy")

          print("==================================")
          print("done vis")

     
          search_instance = silhouette_ksearch(data, 2, 13, algorithm=silhouette_ksearch_type.KMEANS).process()

          amount = search_instance.get_amount()
          scores = search_instance.get_scores()
          print("Scores: '%s'" % str(scores))
          print("amount", amount)
          

          print("==================================")
          print("done scores")

          # xyz = self.pcd[:,0:3]
          # intensity1d = (self.pcd[:,3:4]).flatten()
          # view = pptk.viewer(xyz, intensity1d, type_encoder.get_clusters())
          # view.capture('screenshot.png')
          # print("pptk loaded")
          
    


     def fuzzy_cmeans_clustering(self):
          x = self.pcd
          truth = self.pcd_truth

          print("--------------fuzzy c-means -------------")

          now = datetime.now()
          current_time = now.strftime("%H:%M:%S")
          print("Current Time =", current_time)
         

          fcm = FCM(n_clusters=8)
          fcm.fit(x)

          fcm_centers = fcm.centers
          fcm_labels = fcm.predict(x)

          print("======================")
          print("done clustering")
          now = datetime.now()
          current_time = now.strftime("%H:%M:%S")
          print("Current Time =", current_time)

          print("centers", fcm_centers)
          print("labels", fcm_labels)

          unique_labels = np.unique(fcm_labels)

          # sil = silhouette_score(x, fcm_labels)
          # print("sil", sil)

          print("getting ground truth")
          Clustering.get_ground_truth(self, unique_labels, fcm_labels, truth)
          print("finished getting ground truth")

          # xyz = self.pcd[:,0:3]
          # intensity1d = (self.pcd[:,3:4]).flatten()
          # view = pptk.viewer(xyz, intensity1d, fcm_labels)
          # view.capture('screenshot.png')
          # print("pptk loaded")
          # np.save('/content/drive/Shareddrives/CHSEG/data/fuzzylabels', fcm_labels)

          f, axes = plt.subplots(1, 2, figsize=(11,5))
          axes[0].scatter(x[:,0], x[:,1], alpha=.1)
          axes[1].scatter(x[:,0], x[:,1], c=fcm_labels, alpha=.1)
          axes[1].scatter(fcm_centers[:,0], fcm_centers[:,1], marker="+", s=500, c='w')
          plt.savefig('fuzzy')
          plt.show()


