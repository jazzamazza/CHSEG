import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.mixture import GaussianMixture
from pyclustering.cluster.kmedians import kmedians
from sklearn.metrics import davies_bouldin_score
from fcmeans import FCM
from pyclustering.cluster.silhouette import silhouette_ksearch_type, silhouette_ksearch
from pyclustering.cluster.center_initializer import random_center_initializer
from pyclustering.cluster.encoder import type_encoding, cluster_encoder
from sklearn.metrics import *
import itertools
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture
from sklearn.cluster import KMeans

# testing claass to assess the perfromance of the metrics 
class Testing:
     def __init__(self, pointCloud):
          self.pcd = pointCloud

     def silhouette_kmeans(self):
          
          # x = point cloud 
          x = self.pcd

          print("--------- silhouette k-means ----------")

         # range is 2 to 100 clusters 
          K = range(2, 101)
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
               clusterer = KMeans(n_clusters= k, n_init=10)      #for k-means and k-medoids
               
               cluster_labels = clusterer.fit_predict(x)

               # get silhouette score for k clusters 
               silhouette_avg = silhouette_score(x, cluster_labels)
               print(
                    "For n_clusters =",
                         k,
                    "The average silhouette_score is :",
                         silhouette_avg,
               )
               sample_silhouette_values = silhouette_samples(x, cluster_labels)

               # the rest of the code plots the visual for the silhouette score 
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

     # silhouette score for gaussian mixture model 
     def silhouette_GMM(self):
       
          x = self.pcd

          print("--------- silhouette GMM ----------")

        # choosing range of k between 2 clusters and 100 clusters 
          K = range(2,101)
          for k in K:
              
               # Initialize the clusterer with n_clusters value and a random generato
               clusterer = GaussianMixture(n_components=k, covariance_type='full', random_state=0)
               cluster_labels = clusterer.fit_predict(x)

                # obtain silhouette score 
               silhouette_avg = silhouette_score(x, cluster_labels, metric='euclidean')
               print("{",k,": ", silhouette_avg,"}")

     # silhouette score for fuzzy c-means         
     def silhouette_fuzzy_cmeans(self):
       # x = original point cloud 
        x = self.pcd

        print("--------- silhouette fuzzy c-means ----------")

        # choosing range of k between 2 clusters and 100 clusters 
        K = range(2, 100)
        for k in K:
            
            # initialise FCM, fit the data and get the labels
            fcm = FCM(n_clusters=k)
            fcm.fit(x)
            fcm_labels = fcm.predict(x)

            # obtain the silhouette score 
            sil = silhouette_score(x, fcm_labels)
            print("{",k,":", sil,"}")

     # silhou
     def silhouette_kmedians(self):
       # x = original point cloud 
       x = self.pcd
       print("--------- silhouette k-medians ----------")

        # obtain silhouette scores, from 2 to 100 clusters 
       search_instance = silhouette_ksearch(x, 2, 101, algorithm=silhouette_ksearch_type.KMEDIANS).process()

       amount = search_instance.get_amount()
       scores = search_instance.get_scores()
       print("amount", amount)
       # get silhouette scores 
       print("Scores: '%s'" % str(scores))


     def db_index_kmeans(self):
           # x = original point cloud 
          x = self.pcd

          print("--------- db index k-means ----------")
          results = {}
          
          # range is from 2 to 100 clusters 
          for i in range(2, 101):

               # initialise k-means
               km = KMeans(n_clusters=i, random_state=0)
               # get labels 
               labels = km.fit_predict(x)
               # obtain db index 
               db_index = davies_bouldin_score(x, labels)
               results.update({i: db_index})
               print({i: db_index})


     def db_index_GMM(self):
          # x = original point cloud 
          x = self.pcd

          print("--------- db index GMM ----------")
          results = {}
          
          # range is 2 clusters to 100 clusters 
          for i in range(2,101):
              # intialise GMM
               gm = GaussianMixture(n_components=i, covariance_type='full', random_state=0)
              # obtain labels 
               labels = gm.fit_predict(x)
              # get db index 
               db_index = davies_bouldin_score(x, labels)
               results.update({i: db_index})
               print({i: db_index})

    
     def db_index_fuzzy_cmeans(self):
          # x = original point cloud 
          x = self.pcd

          print("--------- db index fuzzy c-means ----------")
          results = {}
          
          # range is 2 clusters to 100 clusters 
          for i in range(2,101):
              # initialise fFCM 
               fcm = FCM(n_clusters=i)
               fcm.fit(x)
               labels = fcm.predict(x)
               #obtain db index 
               db_index = davies_bouldin_score(x, labels)
               results.update({i: db_index})
               print({i: db_index})
     
     def db_index_Kmedians(self):
          # x = original point cloud 
          x = self.pcd

          print("--------- db index K-medians ----------")
          results = {}
        
        # range is 2 clusters to 100 clusters 
          for i in range(2,101):
              # intialise k-medians 
              initial_medians =  random_center_initializer(x, i).initialize()
              kmedians_instance = kmedians(x, initial_medians)
              clusters = kmedians_instance.get_clusters()


            # get labels 
              type_repr = kmedians_instance.get_cluster_encoding()
              type_encoder = cluster_encoder(type_repr, clusters, x)
              type_encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
              labels_km = type_encoder.get_clusters()

            # obtain db index 
              db_index = davies_bouldin_score(x, labels_km)
              results.update({i: db_index})
              print({i: db_index})

    
     # get BIC score for GMM - adapted from Scikit learn 
     def BIC(self):
          # x = original point cloud 
          x = self.pcd
          lowest_bic = np.infty
          bic = []
          n_components_range = range(1, 70)
          cv_types = ["spherical", "tied", "diag", "full"] #initialise the model with the four different covariance types 
          for cv_type in cv_types:
               for n_components in n_components_range:
               # Fit a Gaussian mixture with EM
                    gmm = mixture.GaussianMixture(
                         n_components=n_components, covariance_type=cv_type, max_iter=600
                    )
                    gmm.fit(x)
                    bic.append(gmm.bic(x))
                    if bic[-1] < lowest_bic:
                         lowest_bic = bic[-1]
                         best_gmm = gmm

          bic = np.array(bic)
          color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
          clf = best_gmm
          bars = []

          # Plot the BIC scores
          plt.figure(figsize=(8, 6))
          spl = plt.subplot(2, 1, 1)
          for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
               xpos = np.array(n_components_range) + 0.2 * (i - 2)
               bars.append(
                    plt.bar(
                     xpos,
                     bic[i * len(n_components_range) : (i + 1) * len(n_components_range)],
                     width=0.2,
                     color=color,
               )
          )

          plt.xticks(n_components_range)
          plt.ylim([bic.min() * 1.01 - 0.01 * bic.max(), bic.max()])
          plt.title("BIC score per model")
          xpos = (
               np.mod(bic.argmin(), len(n_components_range))
               + 0.65
               + 0.2 * np.floor(bic.argmin() / len(n_components_range))
          )

          plt.text(xpos, bic.min() * 0.97 + 0.03 * bic.max(), "*", fontsize=14)
          spl.set_xlabel("Number of components")
          spl.legend([b[0] for b in bars], cv_types)

          # Plot the winner
          splot = plt.subplot(2, 1, 2)
          Y_ = clf.predict(x)
          for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_, color_iter)):
               v, w = linalg.eigh(cov)
               if not np.any(Y_ == i):
                    continue
               plt.scatter(x[Y_ == i, 0], x[Y_ == i, 1], 0.8, color=color)

               # Plot an ellipse to show the Gaussian component
               angle = np.arctan2(w[0][1], w[0][0])
               angle = 180.0 * angle / np.pi  # convert to degrees
               v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
               ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
               ell.set_clip_box(splot.bbox)
               ell.set_alpha(0.5)
               splot.add_artist(ell)

          plt.xticks(())
          plt.yticks(())
          plt.title(
               f"Selected GMM: {best_gmm.covariance_type} model, "
               f"{best_gmm.n_components} components"
          )
          plt.subplots_adjust(hspace=0.35, bottom=0.02)
          plt.show()
          plt.savefig('BIC score.png')
        
    # get evaluation metrics 
     def evaluate(self, metric_choice):
        if metric_choice == 0:
            score = f1_score(self.y_true, self.y_predict, average='macro')
        elif metric_choice == 1:
            # IOU score
            score = jaccard_score(self.y_true, self.y_predict,  average='macro')
        elif metric_choice == 2:
            # precision
            score = precision_score(self.y_true, self.y_predict,  average='macro')
        elif metric_choice == 3:
            # recall
            score = recall_score(self.y_true, self.y_predict, average='macro')
     
        return score
     
  
# method for choosing the desired classification metric
     def classification_metrics(self, actual_ground_truths, predicted_ground_truths):
        # have predicted and actual ground truth for comparison 
        self.y_true = actual_ground_truths 
        self.y_predict = predicted_ground_truths

        print("1",  self.y_true)
        print("2",  self.y_predict)


        score, userInput = "", ""
        while (userInput != "q"):
            userInput = input("\nChoose Classification Metric to Evaluate with:"+
                                    "\n 0 : F1 Score" +
                                    "\n 1 : Intersection Over Union Score"+
                                    "\n 2 : Precision"+
                                    "\n 3 : Recall"+
                                    "\n 4 : Error Rate"+
                                    "\n 5 : Confusion matrix"+
                                    "\n q : Quit\n")
            if (userInput == "q"): break
            score = self.evaluate(int(userInput))
            print(score)
     
if __name__ == "__main__":
    t = Testing()
    t.classification_metrics()


     
