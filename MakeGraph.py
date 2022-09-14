import matplotlib.pyplot as plt
import pandas as pd
def make_plots():
    make_plot("./Results/Done/aggl_all_mixedds_100_750.csv", 
              "Agglomerative clustering on Raw, Geometric, Pointnet++:\nRand Score",
              "Rand score",
              "rand")
    make_plot("./Results/Done/birch_all_mixedds_100_750.csv", 
              "BIRCH clustering on Raw, Geometric, Pointnet++:\nRand Score",
              "Rand score",
              "rand")
    make_plot("./Results/Done/kmeans_all_mixedds_100_750.csv", 
              "K-Means clustering on Raw, Geometric, Pointnet++:\nRand Score",
              "Rand score",
              "rand")
    make_plot("./Results/Done/cure_all_mixedds_100_750.csv", 
              "CURE clustering on Raw, Geometric, Pointnet++:\nRand Score",
              "Rand score",
              "rand")
    
    
    make_plot("./Results/Done/aggl_all_mixedds_100_750.csv", 
              "Agglomerative clustering on Raw, Geometric, Pointnet++:\nDavies-Bouldin score",
              "Davies-Bouldin score",
              "db")
    make_plot("./Results/Done/birch_all_mixedds_100_750.csv", 
              "BIRCH clustering on Raw, Geometric, Pointnet++:\nDavies-Bouldin score",
              "Davies-Bouldin score",
              "db")
    make_plot("./Results/Done/kmeans_all_mixedds_100_750.csv", 
              "K-Means clustering on Raw, Geometric, Pointnet++:\nDavies-Bouldin score",
              "Davies-Bouldin score",
              "db")
    make_plot("./Results/Done/cure_all_mixedds_100_750.csv", 
              "CURE clustering on Raw, Geometric, Pointnet++:\nDavies-Bouldin score",
              "Davies-Bouldin score",
              "db")
    
    
    make_plot("./Results/Done/aggl_all_mixedds_100_750.csv", 
              "Agglomerative clustering on Raw, Geometric, Pointnet++:\nF1 score",
              "F1 score",
              "f1")
    make_plot("./Results/Done/birch_all_mixedds_100_750.csv", 
              "BIRCH clustering on Raw, Geometric, Pointnet++:\nF1 score",
              "F1 score",
              "f1")
    make_plot("./Results/Done/kmeans_all_mixedds_100_750.csv", 
              "K-Means clustering on Raw, Geometric, Pointnet++:\nF1 score",
              "F1 score",
              "f1")
    make_plot("./Results/Done/cure_all_mixedds_100_750.csv", 
              "CURE clustering on Raw, Geometric, Pointnet++:\nF1 score",
              "F1 score",
              "f1")
    
    make_plot("./Results/Done/aggl_all_mixedds_100_750.csv", 
              "Agglomerative clustering on Raw, Geometric, Pointnet++:\nIoU",
              "IoU",
              "jaccard")
    make_plot("./Results/Done/birch_all_mixedds_100_750.csv", 
              "BIRCH clustering on Raw, Geometric, Pointnet++:\nIoU",
              "IoU",
              "jaccard")
    make_plot("./Results/Done/kmeans_all_mixedds_100_750.csv", 
              "K-Means clustering on Raw, Geometric, Pointnet++:\nIoU",
              "IoU",
              "jaccard")
    make_plot("./Results/Done/cure_all_mixedds_100_750.csv", 
              "CURE clustering on Raw, Geometric, Pointnet++:\nIoU",
              "IoU",
              "jaccard")
    
    make_plot("./Results/Done/aggl_all_mixedds_100_750.csv", 
              "Agglomerative clustering on Raw, Geometric, Pointnet++:\nPrecision",
              "Precision",
              "precision")
    make_plot("./Results/Done/birch_all_mixedds_100_750.csv", 
              "BIRCH clustering on Raw, Geometric, Pointnet++:\nPrecision",
              "Precision",
              "precision")
    make_plot("./Results/Done/kmeans_all_mixedds_100_750.csv", 
              "K-Means clustering on Raw, Geometric, Pointnet++:\nPrecision",
              "Precision",
              "precision")
    make_plot("./Results/Done/cure_all_mixedds_100_750.csv", 
              "CURE clustering on Raw, Geometric, Pointnet++:\nPrecision",
              "Precision",
              "precision")
    
    make_plot("./Results/Done/aggl_all_mixedds_100_750.csv", 
              "Agglomerative clustering on Raw, Geometric, Pointnet++:\nRecall",
              "Recall",
              "recall")
    make_plot("./Results/Done/birch_all_mixedds_100_750.csv", 
              "BIRCH clustering on Raw, Geometric, Pointnet++:\nRecall",
              "Recall",
              "recall")
    make_plot("./Results/Done/kmeans_all_mixedds_100_750.csv", 
              "K-Means clustering on Raw, Geometric, Pointnet++:\nRecall",
              "Recall",
              "recall")
    make_plot("./Results/Done/cure_all_mixedds_100_750.csv", 
              "CURE clustering on Raw, Geometric, Pointnet++:\nRecall",
              "Recall",
              "recall")
    
def make_plot(csv_path, title, ylab, metric):
    df = pd.read_csv(csv_path, sep=',', header=0, index_col=0)
    print("Data frame created.")
    pd.set_option("display.max.columns", None)
    print(df.head(5))
    print(df.tail(5))
    
    #make plot
    fig, ax = plt.subplots()
    TITLE = title
    Y_LABEL = ylab
    X_LABEL = "No. clusters"
    #ax.legend() # SET LABELS
    
    # data = df[(df["data_set"] == "cc")]
    # data.plot("n_clusters", "db", ax=ax)
    
    
    labels = {"raw": "Raw Features",
              "cc": "Geometric Features",
              "pnet": "Pointnet++ Features"}
    for data_set in ["raw", "cc", "pnet"]:
        data = df[(df["data_set"] == data_set)]
        data.plot("n_clusters", metric, ax=ax, label=labels[data_set])
    
    #plt.rcParams.update({'font.size': 12})
    ax.set_title(TITLE, {'fontsize': 12})
    ax.set_ylabel(Y_LABEL, {'fontsize': 11})
    ax.set_xlabel(X_LABEL, {'fontsize': 11})
    #plt.title(TITLE, fontsize= 12, fontweight='bold')
    #plt.xlabel(X_LABEL, fontsize=10)
    plt.savefig("./Plots/"+ metric+"_"+ylab+".png")
    plt.show()


if __name__ == "__main__":
    make_plots()