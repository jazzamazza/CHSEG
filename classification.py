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


class Classification:

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

          #sets cluster to majority ground truth  
          g = np.asarray(t)        
          for i in range(0, len(ground_truths)):   #i is each cluster
            print("for")
            if ground_truths[i] == float(1): # if cluster == keep
              for point in t[y_km == i]: # set ground truth of each point to keep
                print("point", point)
                t[y_km == i, 4:5] = float(1)
            else:
              for point in t[y_km == i]:
                print("for 3")
                t[y_km == i, 4:5] = float(0)
                
          print("t shape", np.shape(t))
          print("truth", t[0])
          print("g", g[0])
          print("t", t)

          # for i in unique_labels:
          #     print("cluster:", i)
          #     for point in t[y_km == i]:
          #       print("new point", t[y_km == i, 4:5])

          #np.save('/content/drive/Shareddrives/CHSEG/data/gmm_t_0.5', t)
          print("t[:,4:5].flatten()", t[:,4:5].flatten())
          xyz = self.pcd[:,0:3]
          intensity1d = (self.pcd[:,3:4]).flatten()
          view = pptk.viewer(xyz, intensity1d)
          # view = pptk.viewer(xyz, intensity1d, t[:,4:5].flatten())  #t[:,5:6].flatten()
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
     
    def get_ground_truth_pnet(self, unique_labels, y_km, t):
      
          num_keep, num_discard = 0, 0
          print("t[:,0:3]", t[:,0:3])
          print("t[:,3:4]", t[:,3:4])
    
          ground_truths = np.array([])
          print("t[0]", t[0])
          print("ground_truth size:", ground_truths.size)
          for i in unique_labels:
              num_keep, num_discard = 0, 0
              #print("cluster:", i)
              #for point, p in map(None, x[y_km == i], t[y_km == i]):
              for point in t[y_km == i]:
                #print("p", point[4])
                if (point[3] >= float(0.5)): num_discard += 1
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
          print("len t[:,0:3]", len(t[:,0:3]))
          print("len t[:,3:4] ground truth values", len(t[:,3:4]))
          print("len ground truth", len(ground_truths))
          print("ykm", y_km)

          for i in range(0, len(ground_truths)):   #i is each cluster
            if ground_truths[i] == float(1): # if cluster == keep
              for point in t[y_km == i]: # set ground truth of each point to keep
                t[y_km == i, 3:4] = float(1)
            else:
              for point in t[y_km == i]:
                t[y_km == i, 3:4] = float(0)
                
          print("t shape", np.shape(t))
          print("truth", t[0])
          print("g", g[0])
          # print("t", t)

          # for i in unique_labels:
          #     print("cluster:", i)
          #     for point in t[y_km == i]:
          #       print("new point", t[y_km == i, 4:5])

          # np.save('/content/drive/Shareddrives/CHSEG/data/truth_post_classification', t)
          print("t[:,4:5].flatten()", t[:,4:5].flatten())
          xyz = self.pcd[:,0:3]
          intensity1d = (self.pcd[:,3:4]).flatten()
          view = pptk.viewer(xyz, intensity1d)
          view = pptk.viewer(xyz, intensity1d, t[:,4:5].flatten())  #t[:,5:6].flatten()
          print("pptk loaded")
