import open3d as o3d
import numpy as np
import pptk
from Metrics import Evaluation
from PointCloudLoader import PointCloudLoader
import os
import pandas as pd
import ast

def load(base_dir = "./Data/Clustered/kmeans/", wanted_ds = 0.150):
    with os.scandir(base_dir) as entries:
        for entry in entries:
            if entry.is_file() and entry.path[-4:]=='.ply':
                split_path= entry.path.split('/')[-1:][0][:-4].split("_")
                ds_amt = float(split_path[4])
                alg = split_path[5]
                data_set = split_path[6]
                n_clusters = int(split_path[7])
                #print(""n_clusters, ds_amt)
                if ds_amt == wanted_ds:
                    print(entry.path)
                    path = entry.path
                    print("clustered path:", path)
                   
                    df, csv_path = open_csv(alg, data_set, ds_amt, n_clusters)
                    metrics = df[(df["n_clusters"]==n_clusters)]
                    #metrics_index = df[(df["n_clusters"]==n_clusters)].index
                    cluster_metrics = ast.literal_eval(metrics["clustering_metrics"].values[0])
                    if cluster_metrics != ['db', 'rand', 'sill']:
                        print("init:",cluster_metrics)
                        cluster_metrics.append('sill')
                        print("after:",cluster_metrics)
                    #if metrics["sill"] == 0.0:
                    sill_score = check_sil(path)
                    if 'sill' in df.columns:
                        #metrics["sill"] = sill_score
                        df.loc[df["n_clusters"]==n_clusters, "sill"] = sill_score
                        df.loc[df["n_clusters"]==n_clusters, "clustering_metrics"] = str(cluster_metrics)
                        df.to_csv(csv_path, mode='w', header=True, index=True)
                        print(df.head(2))
                    else:
                        # print("\nloc metrics",metrics_index)
                        df["sill"] = np.zeros(shape=len(df), dtype=float)
                        df.loc[df["n_clusters"]==n_clusters, "sill"] = sill_score
                        df.loc[df["n_clusters"]==n_clusters, "clustering_metrics"] = str(cluster_metrics)
                        df.to_csv(csv_path, mode='w', header=True, index=True)
                        #df.iloc[0]["sill"] = sill_score
                        #df.iloc[0]["clustering_metrics"] = str(cluster_metrics)
                        print(df.head(2))
                        
            
def open_csv(alg, data_set, ds_amt, n_clusters):
    csv_base = "./Results/Done/"
    csv_beg = alg +"_"+ data_set + "_"+str("%.3f" % ds_amt)+"_"
    
    with os.scandir(csv_base) as entries:
        for entry in entries:
            val_find = entry.path.find(csv_beg)
            #print("val find", val_find)
            if entry.is_file() and val_find > -1:
                #print("path:", entry.path)
                split_rest = entry.path[val_find:-4].split("_")
                #print("split rest:", split_rest)
                if int(split_rest[3]) <= n_clusters and int(split_rest[4])>= n_clusters:
                    #print("nclusters",n_clusters,"min clust", split_rest[0],"max_clust",split_rest[1] )
                    path = entry.path
                    print("CSV path:", path)
                    return pd.read_csv(path, sep=',', header=0, index_col=0), path
                    
                
    
def check_sil(path):
    #path = load()
    #path = "./Data/Clustered/cure/church_registered_clusters_ds_0.35_cure_pnet_158_2022-09-13_19_39PM.ply"
    pcd = o3d.io.read_point_cloud(path, print_progress=True)
    in_file = path.split('/')[-1:][0]
    # in_file = in_file[-1:]
    print(in_file)
    info_list = in_file[:-4].split("_")
    index = 0
    while (info_list[index].find("0.") == -1):
        if info_list[index]== "clusters":
            idx_clust = index
        index += 1

    ds_amt = str("%.3f" % float(info_list[index]))
    alg = info_list[index+1]
    data_set = info_list[index+2]

    if data_set == "pnet":
        path_append = "_all_fix"
        ext = ".npy"
        root_path = "./Data/PNet/"
    elif data_set == "cc":
        path_append = "_23_feats"
        ext = ".las"
        root_path = "./Data/CC/"
    else:
        path_append = ""
        ext = ".npy"
        root_path = "./Data/"
    join_list = []+ info_list[0:idx_clust] + info_list[idx_clust+1:index]
    print(join_list)
    find_file = ('_'.join(join_list[0:len(join_list)])) + "_" + ds_amt + "_" + data_set + path_append + ext
    print("findfile:", find_file)  
        
    find_path = root_path + find_file
    loader = PointCloudLoader(find_path)
    orig_pcd, orig_pcd_all = loader.load_point_cloud()
    print("pcd:",np.shape(orig_pcd),"pcd all:",np.shape(orig_pcd_all))


    points = np.asarray(pcd.points)
    clusters = np.asarray(pcd.normals)[:, 1:2]
    u_clusters = np.unique(clusters)
    n_clusters = len(u_clusters)
    print("clusters 0 - 9:", u_clusters[0:10])
    print("n clusters:", n_clusters)
    truth = np.asarray(pcd.colors)[:, 0:1]
    u_true = np.unique(truth)
    print(u_true)
    pred = np.asarray(pcd.colors)[:, 1:2]

    eval = Evaluation(truth)
    sill = eval.evaluate_clusters(y_true= truth, y_pred= pred, cluster_labels= clusters, input_pcd= orig_pcd, metric_choice="sill")
    print("sill", sill)
    return sill
    
def find_best(metric, min_max, data_set, alg, ds_amt):
    base_path = "./Results/Done/"
    file_name = alg+"_"+data_set+"_"+ds_amt+"_100_750.csv"
    path = base_path+file_name
    
    df = pd.read_csv(path)
    if min_max == 'min':
        min_max_metric = min(df[metric])
    elif min_max == 'max':
        min_max_metric = max(df[metric])
    
    n_clusters = df.loc[df[metric]==min_max_metric]
    print(n_clusters.values)
    n_clusters = n_clusters["n_clusters"]
    print("min/max:", min_max_metric)
    print("at n clusters:", n_clusters)
        
if __name__ == "__main__":
    load()
    # dsets =["raw", "cc", "pnet"]
    # min_db_cure = ["db", "min", "raw", "cure", "0.195"]
    # for dset in dsets:
    #     min_db_cure[2] = dset
    #     find_best(*min_db_cure)