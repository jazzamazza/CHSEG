from sklearnex import patch_sklearn
patch_sklearn()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import faiss
import pptk


from sklearn_extra.cluster import KMedoids

#jared methods
from sklearn.cluster import Birch
from pyclustering.cluster.cure import cure
from pyclustering.cluster.rock import rock

from pyclustering.cluster import cluster_visualizer_multidim
from pyclustering.utils import read_sample
from pyclustering.cluster import cluster_visualizer

#jemma methods
from sklearn.cluster import AffinityPropagation

from PointCloudUtils import PointCloudUtils

import open3d as o3d



# Clustering class with various clustering methods
class Clustering:
     def __init__(self, pointCloud, pcd_choice, pcd_truth):
          self.pcd = pointCloud
          self.pcd_truth = pcd_truth
          if (pcd_choice == "1"): self.type = "raw"
          elif (pcd_choice == "2"): self.type = "cldCmp"
          elif (pcd_choice == "3"): self.type = "pnet++"
     
     # K-MEANS CLUSTERING USING FAISS LIBRARY - SPEEDS UP COMPUTATION
     def k_means_clustering_faiss(self, k, imageName):
      x = self.pcd
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
          
          # Visualise K-Means
          y_km = kmeans.predict(x)
          fpred = kmeans.fit_predict(x)
          print("y_km:", np.shape(y_km))
          centroids = kmeans.cluster_centers_
          print("centroids:", np.shape(centroids))
          unique_labels = np.unique(y_km)
          print("unique_labels:", unique_labels)
          
          num_keep, num_discard = 0, 0
    
          ground_truths = np.array([])
          print("ground_truth size:", ground_truths.size)
          for i in unique_labels:
              num_keep, num_discard = 0, 0
              print("**** cluster ****\n->", i)
              # for point, p in map(None, x[y_km == i], t[y_km == i]):
              for point in t[y_km == i]:
                print("p", point[4])
                if (point[4] >= float(0.5)): 
                    num_discard += 1
                else: 
                    num_keep += 1
              print("num_keep:", num_keep)
              print("num_discard:", num_discard)
              if num_keep > num_discard: 
                ground_truths = np.append(ground_truths, 1)
              else: 
                ground_truths = np.append(ground_truths, 0)

          print("ground_truth:", ground_truths)
          
          # keep = np.asarray([])
          # discard = np.asarray([])
          for i in range(0, len(ground_truths)):   #i is each cluster - changing points in t
            if ground_truths[i] == float(1):
              for point in t[y_km == i]:
                point[4] = float(1)
            else:
              for point in t[y_km == i]:
                point[4] = float(0)
          
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

          print("t", t)
          print("y_km", y_km)

          xyz = self.pcd[:,0:3]
          intensity1d = (self.pcd[:,3:4]).flatten()
          view = pptk.viewer(xyz, intensity1d, t[:,5:6].flatten())
          print("pptk loaded")

          # num_keep, num_discard = 0
          # groundTruths = []
          # for i in unique_labels: 
          #      x[y_km == i , 0] , x[y_km == i , 1]          #prints out the points in each label/cluster 
          #           index x,y,z values into the raw data set --> get the ground truth label
          #      if groundTruths == 1: num_keep += 1
          #      else: num_discard += 1
          #      if num_keep > num_discard: 
          #           groundTruths[i] = 1
          #      else: groundTruths[i] = 0
          
          return y_km, fpred
          # intensity_1d = x[:,3:4].flatten()
          # points = x[:,:3]
          # print("Visualising in PPTK")
          # # intensity_1d = intensity.flatten()
          # # truth_label_1d = truth_label.flatten()
          # view = pptk.viewer(points,intensity_1d, pred_lab)
          # print("PPTK Loaded")
          
          # unique_labels = np.unique(y_km)
          # print("unique_labels:", unique_labels)
          # for i in unique_labels:
          #      plt.scatter(x[y_km == i , 0] , x[y_km == i , 1] , label = i, marker='o', picker=True)
          # plt.scatter(
          #      centroids[:, 0], centroids[:, 1],
          #      s=100, marker='*',
          #      c='red', edgecolor='black',
          #      label='centroids'
          # )
          # #plt.legend()
          # plt.title('K-Means Clustering')
          # plt.savefig('k_means_clusters.png') 
          # plt.show()
          
     def find_quality(self):
          pcutils = PointCloudUtils()
          pcloud = self.pcd
          pcutils.get_attributes(pcloud, "pcloud")
          pcloud_len = np.shape(pcloud)[0]
          points = pcloud[:,:3]
          intensity = pcloud[:,3:4] 
          #gtruth = pcloud[:,4:5]
          
          # format using open3d
          pcd = o3d.geometry.PointCloud()
          pcd.points = o3d.utility.Vector3dVector(points) # add {x,y,z} points to pcd
          intensity_to_rgb = np.hstack((intensity, np.zeros((pcloud_len,1)) , np.zeros((pcloud_len,1)))) # form a 3D vector to add to o3d pcd
          pcutils.get_attributes(intensity_to_rgb, "intrgb")
          pcd.colors = o3d.utility.Vector3dVector(intensity_to_rgb) # store intensity as every value in color vector
          
          clusters, pred = self.k_means_clustering(13)
          clusters = np.reshape(clusters,(-1,1))
          pred = np.reshape(pred,(-1,1))
          
          pcloud = np.load("./Data/church_registered_raw_0.5.npy")
          gtruth = pcloud[:,4:5]
          pcloud = np.load("./Data/ground_truth.npy")
          print("56",pcloud[:,5:6])
          keepdiscard = pcloud[:,5:6]
          
          pcutils.get_attributes(keepdiscard)
          pcutils.get_attributes(gtruth, "gtruth")
          pcutils.get_attributes(clusters, "clusters")
          pcutils.get_attributes(pred, "pred")
          
          gtruth_clust_to_normal = np.hstack((gtruth, clusters, keepdiscard))
          pcutils.get_attributes(gtruth_clust_to_normal, "gtclust")
          pcd.normals = o3d.utility.Vector3dVector(gtruth_clust_to_normal) #store keep discard as 
          
          outpcloud = np.hstack(((np.asarray(pcd.points)), (np.asarray(pcd.colors)), (np.asarray(pcd.normals))))
          pcutils.get_attributes(outpcloud, "outpcloud")
          print(pcd)
          
          output_path = "./Data/church_registered_kmeans_"+str(0.05)
          np.save(output_path + ".npy", outpcloud)
          print(o3d.io.write_point_cloud(output_path+".ply", pcd, print_progress=True))       
          print("done")
     
     
     def birch(self, k):
          heading = "BIRCH Clustering"
          heading = ('*' * len(heading)) + heading + ('*' * len(heading))
          print(heading)
          print("Using", k, "Clusters")
          birch = Birch(n_clusters=k)
          x = self.pcd
          print("X shape",np.shape(x))
          print("Fit start")
          birch.fit(x)
          print("Pred start")
          pred_lab = birch.predict(x)
          print("labels",pred_lab)
          print("shape",np.shape(pred_lab))
          intensity_1d = x[:,3:4].flatten()
          points = x[:,:3]
          print("Visualising in PPTK")
          # intensity_1d = intensity.flatten()
          # truth_label_1d = truth_label.flatten()
          view = pptk.viewer(points,intensity_1d, pred_lab)
          print("PPTK Loaded")
          
          unique_labels = np.unique(pred_lab)
          print("unique_labels:", unique_labels)
          for i in unique_labels:
               plt.scatter(x[pred_lab == i , 0] , x[pred_lab == i , 1] , label = i, marker='o', picker=True)
          # plt.scatter(
          #      centroids[:, 0], centroids[:, 1],
          #      s=100, marker='*',
          #      c='red', edgecolor='black',
          #      label='centroids'
          # )
          #plt.legend()
          plt.title('K-Means Clustering')
          plt.savefig('k_means_clusters.png') 
          plt.show()
          
     def cure_clustering(self, k):
          k=3
          heading = "CURE Clustering"
          heading = ('*' * len(heading)) + heading + ('*' * len(heading))
          print(heading)
          print("Using", k, "Clusters")
          x = np.asarray(self.pcd)
          print("PCD shape",np.shape(x))
          cure_cluster = cure(x,k, ccore=True)
          print("process start")
          cure_cluster.process()
          print("process end")
          #print("cluster enc", cure_cluster.get_cluster_encoding())
          clusters = cure_cluster.get_clusters()
          print("shape clust",np.shape(clusters))
          # pcutil = PointCloudUtils()
          # pcutil.get_attributes(clusters, "CLUSTERS")
          means = cure_cluster.get_means()
          print("shape means",np.shape(means))
          # pcutil.get_attributes(means, "MEANS")
          reps = cure_cluster.get_representors()
          # pcutil.get_attributes(reps, "REPRESENTORS")
          print("shape reps 0",np.shape(reps))
          #print("reps", reps[0])
          
          visualizer = cluster_visualizer_multidim()

          #visualizer.append_clusters(clusters, x)
          
          # flat_list=[]
          # for sublist in clusters:
          #      for item in sublist:
          #           flat_list.append(item)
          
          visualizer.append_clusters(clusters, x.tolist(), marker = 'o', markersize= 5)
          #visualizer.append_cluster(means, x.tolist(), '*', 5)
          visualizer.show()
               

          # visualizer.show()
          
          
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
          plt.title('2 clusters of data: K-mediods')
          plt.savefig('k_mediods_clusters.png') 
          plt.show()
#To overcome the problem of sensitivity to outliers, instead of taking the mean value 
# as the centroid, we can take actual data point to represent the cluster, this is what K-medoids does.

# try visualize
# then try change covariance to diag and spherical - important estimator
          
     def affinity_progpogation_clustering(self):
          #the reason it crashes is due to size of point cloud - needs to be downsampled heavily 
          
          print("\n------------------ Affinity Propogation ---------------------")
          
          x = self.pcd
          

          # K = 10
          # j = np.array(arr[0])
          # print("first size", j.size)

          # for k in range(1, K):
          #      print(arr[k])
          #      np.append(j, arr[k])
          #      print("in loop size", j.size)
               
          # print("j[0]", j[0])
          # print("size", j.size)
          # x = np.reshape(j, (-1,2))
          # print("x[0]", x[0])
          # print("x size", x.size)
          
          #init = (-1)*(np.max(pdist(x))*np.max(pdist(x)))
          #init = -1*np.max(pdist(x))
          print("HI")
          clustering = AffinityPropagation(damping = 0.5, random_state=5).fit(x) #crashes here 
          print("HELLO")

          cluster_centers_indices = clustering.cluster_centers_indices_
          n_clusters_ = len(cluster_centers_indices)
          print("Estimated number of clusters: %d" % n_clusters_)
          
          labels = clustering.labels_
          print("labels", labels)
          
          x  = x.copy(order='C')
          y_km = clustering.predict(x)

          centroids = clustering.cluster_centers_
          print("centroids:", centroids)



          unique_labels = np.unique(y_km)
          print("unique_labels:", unique_labels)

          xyz = self.pcd[:,0:3]
          intensity1d = (self.pcd[:,3:4]).flatten()
          view = pptk.viewer(xyz, intensity1d, y_km)
          print("pptk loaded")


          for i in unique_labels:
               plt.scatter(x[y_km == i , 0] , x[y_km == i , 1] , label = i, marker='o', picker=True)
          plt.scatter(
               centroids[:, 0], centroids[:, 1],
               s=100, marker='*',
               c='red', edgecolor='black',
               label='centroids'
          )
          #gives 0.46 for sqeuclidean
          print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(x, labels, metric="sqeuclidean"))
          #gives 0.3 for  euclidean 
          S = metrics.silhouette_score(x, labels, metric='euclidean', sample_size=None, random_state=None)
          print("s", S)
          
          
          # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
          # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
          # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
          # print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
          # print(
          #      "Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels)
          # )

          plt.title('affinity propogation')
          plt.savefig('affinity_propogation.png') 
          plt.show()     

     def silhouette(self):
          x = self.pcd
          K = range(2, 70)
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
               clusterer = Birch(n_clusters= k)
               cluster_labels = clusterer.fit_predict(x)

               silhouette_avg = silhouette_score(x, cluster_labels)
               print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
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

               ax1.set_yticks([])  # Clear the y-axis labels / ticks
               ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

               # 2nd Plot showing the actual clusters formed
               colors = cm.nipy_spectral(cluster_labels.astype(float) / k)
               ax2.scatter(x[:, 0], x[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")

               # Labeling the clusters
               # centers = clusterer.
               # # Draw white circles at cluster centers
               # ax2.scatter(
               #      centers[:, 0],
               #      centers[:, 1],
               #      marker="o",
               #      c="white",
               #      alpha=1,
               #      s=200,
               #      edgecolor="k",
               # )

               # for i, c in enumerate(centers):
               #      ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

               # ax2.set_title("The visualization of the clustered data.")
               # ax2.set_xlabel("Feature space for the 1st feature")
               # ax2.set_ylabel("Feature space for the 2nd feature")

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
