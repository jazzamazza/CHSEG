import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import faiss

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

          print("ground truth in clustering:", t[:,4:5])
          
          # Visualise K-Means
          y_km = kmeans.predict(x)
          print("y_km:", y_km)
          print("arrays EQUAL?",np.array_equal(y_km, x))
          centroids = kmeans.cluster_centers_
          print("centroids:", centroids)
          unique_labels = np.unique(y_km)
          print("unique_labels:", unique_labels)
    
          self.classification(unique_labels, y_km, t)
               
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
          plt.savefig('k_means_clusters.png') 
          plt.show()

     # Classification
     def classification(self, unique_labels, y_km, pcd_with_truth):
        t = pcd_with_truth
        ground_truths = np.array([])
        print("ground_truth size:", ground_truths.size)
        for i in unique_labels:
            num_keep, num_discard = 0, 0
            print("cluster:", i)
            for point in t[y_km == i]:
                print("p", point[4])
                if (point[4] >= float(0.5)): num_discard += 1
                else: num_keep += 1
            print("num_keep:", num_keep)
            print("num_discard:", num_discard)
            if num_keep > num_discard: 
                ground_truths = np.append(ground_truths, 0)
            else: 
                ground_truths = np.append(ground_truths, 1)
        print("ground_truth:", ground_truths)

        g = np.asarray(t)
        for i in range(0, len(ground_truths)):   #i is each cluster
            if ground_truths[i] == float(1): # if cluster == keep
                for point in t[y_km == i]: # set ground truth of each point to keep
                    t[y_km == i, 4:5] = float(1)
            else:
                for point in t[y_km == i]:
                    t[y_km == i, 4:5] = float(0)
        print("t shape", np.shape(t))
        print("t[0]", t[0])

        for i in unique_labels:
            print("cluster:", i)
            for point in t[y_km == i]:
                print("new point", t[y_km == i, 4:5])

        np.save('/content/drive/Shareddrives/Thesis/Data/ground_truth_new', t)
            
