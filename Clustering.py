from sklearnex import patch_sklearn

patch_sklearn()
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# jared methods
from sklearn.cluster import Birch
from pyclustering.cluster.cure import cure
from pyclustering.cluster.rock import rock
from pyclustering.cluster import cluster_visualizer_multidim
from pyclustering.cluster.encoder import cluster_encoder
from pyclustering.cluster.encoder import type_encoding
from pyclustering.utils import read_sample
from pyclustering.cluster import cluster_visualizer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pptk
import open3d as o3d
from tqdm import tqdm
from PointCloudUtils import PointCloudUtils

# Clustering class with various clustering methods
class Clustering:
    def __init__(self, point_cloud, pcd_truth, pcd_choice):
        self.pcd = point_cloud
        self.pcd_truth = pcd_truth
        # self.pcd_truth = pcd_with_truths
        if pcd_choice == "1":
            self.type = "raw"
        elif pcd_choice == "2":
            self.type = "cldCmp"
        elif pcd_choice == "3":
            self.type = "pnet++"

    def get_ground_truth(self, unique_labels, y_km, t):

        if np.shape(self.pcd_truth)[1] > 5:
            index_truth = np.shape(self.pcd_truth)[1]  # last 1d
        else:
            index_truth = 4

        num_keep, num_discard = 0, 0

        ground_truths = np.array([])
        print("t[0]", t[0])
        print("ground_truth size:", ground_truths.size)
        for i in tqdm(unique_labels, desc="quick loop"):
            num_keep, num_discard = 0, 0
            # print("cluster:", i)
            # for point, p in map(None, x[y_km == i], t[y_km == i]):
            for point in t[y_km == i]:
                # print("p", point[4])
                if point[index_truth - 1] >= float(0.5):
                    num_discard += 1
                else:
                    num_keep += 1
            # print("num_keep:", num_keep)
            # print("num_discard:", num_discard)
            if num_keep > num_discard:
                ground_truths = np.append(
                    ground_truths, 1
                )  # changing the clusters to keep and discard
            else:
                ground_truths = np.append(ground_truths, 0)

        print("ground_truth:", ground_truths)

        # sets cluster to majority ground truth
        g = np.asarray(t)
        for i in tqdm(
            range(0, len(ground_truths)), desc="for loops"
        ):  # i is each cluster
            # print("for")
            if ground_truths[i] == float(1):  # if cluster == keep
                for point in tqdm(
                    t[y_km == i], desc="set keep loop"
                ):  # set ground truth of each point to keep
                    t[y_km == i, (index_truth - 1) : index_truth] = float(
                        0
                    )  # 0 in pcloud is keep
            else:
                for point in tqdm(t[y_km == i], desc="set discard loop"):
                    t[y_km == i, (index_truth - 1) : index_truth] = float(
                        1
                    )  # 1 in pcloud is discard

        print("t shape", np.shape(t))
        print("truth", t[0])
        print("g", g[0])
        print("t", t)
        print("t[:,4:5].flatten()", t[:, (index_truth - 1) : index_truth].flatten())

        np.save("./Data/truths_class.npy", t)
        tflat = t.flatten()
        truthorig = self.pcd_truth[:, index_truth - 1 : index_truth].flatten()
        xyz = self.pcd[:, 0:3]
        intensity1d = (self.pcd[:, 3:4]).flatten()
        view = pptk.viewer(xyz, intensity1d, truthorig, tflat)

        print("pptk loaded")

    def k_means_clustering(self, k):
        x = self.pcd
        t = self.pcd_truth

        print("\n------------------k means---------------------")
        kmeans = KMeans(n_clusters=k, n_init=10)  # number of clusters (k)
        kmeans.fit(x)  # apply k means to dataset
        print("x[0]", x[0])
        # Visualise K-Means
        y_km = kmeans.predict(x)
        print("y_km:", y_km)  # 10 clusters

        centroids = kmeans.cluster_centers_
        print("centroids:", centroids)
        unique_labels = np.unique(y_km)
        print("unique_labels:", unique_labels)

        # get ground truth
        print("t in kmeans", t[0])
        self.get_ground_truth(unique_labels, y_km, t)
        print("get ground truth")

        for i in unique_labels:
            # print((x[y_km == i , 0] , x[y_km == i , 1])) #how to access the pointd
            plt.scatter(
                x[y_km == i, 0], x[y_km == i, 1], label=i, marker="o", picker=True
            )
        plt.scatter(
            centroids[:, 0],
            centroids[:, 1],
            s=100,
            marker="*",
            c="red",
            edgecolor="black",
            label="centroids",
        )
        # plt.legend()
        plt.title("Two clusters of data")
        plt.savefig("k_means_clusters.png")
        plt.show()

    def find_quality(self):
        pcutils = PointCloudUtils()
        pcloud = self.pcd
        pcutils.get_attributes(pcloud, "pcloud")
        pcloud_len = np.shape(pcloud)[0]
        points = pcloud[:, :3]
        intensity = pcloud[:, 3:4]
        # gtruth = pcloud[:,4:5]

        # format using open3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)  # add {x,y,z} points to pcd
        intensity_to_rgb = np.hstack(
            (intensity, np.zeros((pcloud_len, 1)), np.zeros((pcloud_len, 1)))
        )  # form a 3D vector to add to o3d pcd
        pcutils.get_attributes(intensity_to_rgb, "intrgb")
        pcd.colors = o3d.utility.Vector3dVector(
            intensity_to_rgb
        )  # store intensity as every value in color vector

        clusters, pred = self.k_means_clustering(13)
        clusters = np.reshape(clusters, (-1, 1))
        pred = np.reshape(pred, (-1, 1))

        pcloud = np.load("./Data/church_registered_raw_0.5.npy")
        gtruth = pcloud[:, 4:5]
        pcloud = np.load("./Data/ground_truth.npy")
        print("56", pcloud[:, 5:6])
        keepdiscard = pcloud[:, 5:6]

        pcutils.get_attributes(keepdiscard)
        pcutils.get_attributes(gtruth, "gtruth")
        pcutils.get_attributes(clusters, "clusters")
        pcutils.get_attributes(pred, "pred")

        gtruth_clust_to_normal = np.hstack((gtruth, clusters, keepdiscard))
        pcutils.get_attributes(gtruth_clust_to_normal, "gtclust")
        pcd.normals = o3d.utility.Vector3dVector(
            gtruth_clust_to_normal
        )  # store keep discard as

        outpcloud = np.hstack(
            (
                (np.asarray(pcd.points)),
                (np.asarray(pcd.colors)),
                (np.asarray(pcd.normals)),
            )
        )
        pcutils.get_attributes(outpcloud, "outpcloud")
        print(pcd)

        output_path = "./Data/church_registered_kmeans_" + str(0.05)
        np.save(output_path + ".npy", outpcloud)
        print(o3d.io.write_point_cloud(output_path + ".ply", pcd, print_progress=True))
        print("done")

    def birch(self, k):
        heading = "BIRCH Clustering"
        heading = ("*" * len(heading)) + heading + ("*" * len(heading))
        print(heading)
        print("Using", k, "Clusters")
        birch = Birch(n_clusters=k)
        x = self.pcd
        print("X shape", np.shape(x))
        print("Fit start")
        birch.fit(x)
        print("Pred start")
        pred_lab = birch.predict(x)
        print("labels", pred_lab)
        print("shape", np.shape(pred_lab))
        intensity_1d = x[:, 3:4].flatten()
        points = x[:, :3]
        print("Visualising in PPTK")
        # intensity_1d = intensity.flatten()
        # truth_label_1d = truth_label.flatten()
        view = pptk.viewer(points, intensity_1d, pred_lab)
        print("PPTK Loaded")

        unique_labels = np.unique(pred_lab)
        print("unique_labels:", unique_labels)
        for i in unique_labels:
            plt.scatter(
                x[pred_lab == i, 0],
                x[pred_lab == i, 1],
                label=i,
                marker="o",
                picker=True,
            )
        # plt.scatter(
        #      centroids[:, 0], centroids[:, 1],
        #      s=100, marker='*',
        #      c='red', edgecolor='black',
        #      label='centroids'
        # )
        # plt.legend()
        plt.title("K-Means Clustering")
        plt.savefig("k_means_clusters.png")
        plt.show()

    def cure_clustering(self, k=3):
        clustering_alg = "CURE Clustering"
        decoration = "*" * len(clustering_alg)
        heading = decoration + clustering_alg + decoration
        print(heading)

        data = np.asarray(self.pcd)
        pcutils = PointCloudUtils()
        pcutils.get_attributes(data, "Input PCD")

        # to do num rep_points, compression
        cure_cluster = cure(data, k, ccore=True)
        print("Starting using", k, "clusters...")
        cure_cluster.process()
        print("Clustering finished")

        clusters = cure_cluster.get_clusters()
        means = cure_cluster.get_means()
        reps = cure_cluster.get_representors()

        encoding = cure_cluster.get_cluster_encoding()
        print("Encoding:", encoding)
        encoder = cluster_encoder(encoding, clusters, data)
        encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
        print("New Encoding:", encoder.get_encoding())
        clusters = encoder.get_clusters()
        clusters1d = clusters.flatten()

        view = pptk.viewer(data[:, :3], clusters1d)

        # visualizer = cluster_visualizer_multidim()
        # visualizer.append_clusters(clusters, x)
        # flat_list=[]
        # for sublist in clusters:
        #      for item in sublist:
        #           flat_list.append(item)
        # visualizer.append_clusters(clusters, x.tolist(), marker="o", markersize=5)
        # visualizer.append_cluster(means, x.tolist(), '*', 5)
        # visualizer.show()

    def silhouette(self):
        x = self.pcd
        K = range(2, 70)
        for k in K:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

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
                if point[4] >= float(0.5):
                    num_discard += 1
                else:
                    num_keep += 1
            print("num_keep:", num_keep)
            print("num_discard:", num_discard)
            if num_keep > num_discard:
                ground_truths = np.append(ground_truths, 0)
            else:
                ground_truths = np.append(ground_truths, 1)
        print("ground_truth:", ground_truths)

        g = np.asarray(t)
        for i in range(0, len(ground_truths)):  # i is each cluster
            if ground_truths[i] == float(1):  # if cluster == keep
                for point in t[y_km == i]:  # set ground truth of each point to keep
                    t[y_km == i, 4:5] = float(1)
            else:
                for point in t[y_km == i]:
                    t[y_km == i, 4:5] = float(0)
        print("t shape", np.shape(t))
        print("t[0]", t[0])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = Birch(n_clusters=k)
        cluster_labels = clusterer.fit_predict(x)
        for i in unique_labels:
            print("cluster:", i)
            for point in t[y_km == i]:
                print("new point", t[y_km == i, 4:5])

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
                    ith_cluster_silhouette_values = sample_silhouette_values[
                        cluster_labels == i
                    ]
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
                ax2.scatter(
                    x[:, 0],
                    x[:, 1],
                    marker=".",
                    s=30,
                    lw=0,
                    alpha=0.7,
                    c=colors,
                    edgecolor="k",
                )

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

    # output
    # For n_clusters = 2 The average silhouette_score is : 0.3889045527426348
    # For n_clusters = 3 The average silhouette_score is : 0.3403178007585951
    # For n_clusters = 4 The average silhouette_score is : 0.31642498946572917
    # For n_clusters = 5 The average silhouette_score is : 0.30732018048203114
    # For n_clusters = 6 The average silhouette_score is : 0.32511128283087165
    # For n_clusters = 7 The average silhouette_score is : 0.326443764304626
    # For n_clusters = 8 The average silhouette_score is : 0.329213106052044
    # For n_clusters = 9 The average silhouette_score is : 0.31500958433214615
