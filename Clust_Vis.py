## VISUALISE CLUSTERING WITH MATPLOTLIB:

import matplotlib.pyplot as plt
from Outputting import img_path
import numpy as np
from itertools import cycle

class Clust_Vis():

    def vis_k_means(self, unique_labels, centroids, y_km, x): 
        for i in unique_labels:
             plt.scatter(x[y_km == i , 0] , x[y_km == i , 1] , label = i, marker='o', picker=True)
        plt.scatter(
             centroids[:, 0], centroids[:, 1],
             s=100, marker='*',
             c='red', edgecolor='black',
             label='centroids'
        )
        plt.title('K-Means Clustering')
        plt.savefig(img_path + 'k_means_clusters.png') 
        plt.show()
    
    def vis_DBSCAN(self, X, y_db, db, core_samples_mask):
        # visualise
        imgName = img_path + 'DBSCAN_clusters_' + self.type + '.png'
        self.visualiseClusters("DBSCAN-Shift Clustering", X, y_db, imgName)
        
        # visualise 2
        imgName = img_path + 'DBSCAN_clusters2_' + self.type + '.png'
        self.visualiseClusters2("DBSCAN-Shift Clustering2", X, y_db, imgName, db)
        
         # visualise 4
        imgName = img_path + "DBSCAN_clusters1_" + self.type + '.png'
        self.visualiseClusters4("DBSCAN Clustering 4", X, y_db, imgName, core_samples_mask)
        
    def vis_OPTICS(self, X, reachability, y_op):
        # Reachability plot
        space = np.arange(len(X))
        colors = ["g.", "r.", "b.", "y.", "c."]
        for klass, color in zip(range(0, 5), colors):
            Xk = space[y_op == klass]
            Rk = reachability[y_op == klass]
            plt.plot(Xk, Rk, color, alpha=0.3)
            plt.plot(space[y_op == -1], reachability[y_op == -1], "k.", alpha=0.3)
            plt.plot(space, np.full_like(space, 2.0, dtype=float), "k-", alpha=0.5)
            plt.plot(space, np.full_like(space, 0.5, dtype=float), "k-.", alpha=0.5)
            plt.title("Reachability (epsilon distance)")
            plt.ylabel("Reachability Plot")
        plt.savefig(img_path + "OPTICS_reachability.png")
        plt.show()
        
        # OPTICS
        colors = ["g.", "r.", "b.", "y.", "c."]
        for klass, color in zip(range(0, 5), colors):
            print("color", color)
            Xk = X[y_op == klass]
            plt.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
            plt.plot(X[y_op == -1, 0], X[y_op == -1, 1], "k+", alpha=0.1)
            plt.title("Automatic Clustering\nOPTICS")
        plt.tight_layout()
        plt.savefig(img_path + "Optics_clusters1_" + self.type + '.png')
        plt.show()

        # visualise
        imgName = img_path + 'Optics_clusters_' + self.type + '.png'
        self.visualiseClusters("Optics Clustering", X, y_op, imgName)
    
    def vis_mean_shift(self, X, y_ms, ms):
        # visualise 1
        imgName = img_path + 'Mean-Shift_clusters1_' + self.type + '.png'
        self.visualiseClusters("Mean-Shift Clustering1", X, y_ms, imgName)

        # visualise 2
        imgName = img_path + 'Mean-Shift_clusters2_' + self.type + '.png'
        predict = ms.predict(X)
        self.visualiseClusters2("Mean-Shift Clustering2", X, y_ms, imgName, predict)

        # visualise 3
        imgName = img_path + "Mean-Shift_clusters3_" + self.type + ".png"
        self.visualiseClusters3("Mean-Shift Clustering3", X, y_ms, imgName, ms.cluster_centers_)
    
    def vis_elbow_method(self, distances):
        plt.plot(distances)
        plt.xlabel("Points")
        plt.ylabel("Distance")
        plt.savefig(img_path + 'DBSCAN_Eps.png')
        plt.show()

        plt.xlabel("Points")
        plt.ylabel("Distance")
        plt.savefig(img_path + 'DBSCAN_elbow.png')
        plt.show()
        

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
