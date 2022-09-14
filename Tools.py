# from datetime import date
import importlib
import os
import sys
from PointCloudUtils import PointCloudUtils
import numpy as np
import pptk
from Clustering import Clustering
from Metrics import ClusterMetrics
import pandas as pd

class Tools:
    def __init__(self):
        self.pcutils = PointCloudUtils()

    def run_pnet(self, in_path, ds_amt):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ROOT_DIR = BASE_DIR
        sys.path.append(os.path.join(ROOT_DIR, "PointNet++"))
        pnet = importlib.import_module("test_semseg")
        return pnet.main_semseg(in_path, ds_amt)

    def view_pnet(self, pnet_cloud = None, file = None):
        if pnet_cloud is None:
            pnet_cloud = np.load(file)
        points = pnet_cloud[:,:3]
        truth = pnet_cloud[:,3:4]
        feats = pnet_cloud[:,4:]
        feat_1 = feats[:,:1]
        feat_2 = feats[:,1:2]
        feat_3 = feats[:,2:3]
        #print(np.shape(feats[:,:1]))
        viewer = pptk.viewer(points, truth.flatten(), feat_1.flatten(), feat_2.flatten(), feat_3.flatten(), debug = True)
        viewer.wait()
        viewer.close()
        
    def fix_pnet(self, file):
        pnet_cloud = np.load(file)
        self.pcutils.get_attributes(pnet_cloud)
        points = pnet_cloud[:,:3]
        truth = pnet_cloud[:,3:4]
        feats = pnet_cloud[:,4:]
        unique_points, unique_point_indicies = np.unique(points, axis = 0, return_index=True)
        unique_truth = truth[unique_point_indicies]
        unique_feats = feats[unique_point_indicies]
        print("len input points:",len(points.flatten()),"\nlen unique points:",len(unique_points))
        out_pcd = np.hstack((unique_points, unique_truth, unique_feats))
        name = file[:-4] + "_fix.npy"
        np.save(name, out_pcd)
        
    def make_pnet(self, file_path, is_ds, ds_amt):
        #pcd = np.load("./Data/PNet/church_registered_ds_0.075_0.085_pnet.npy")
        self.pcutils.npy_to_pnet(file_path, is_ds, float(ds_amt))

    def menu(self):
        print("Welcome to Tools")
        menu_selection = input(
            "\nPlease select an option from the menu:"
            + "\n1.) Auto Downsample"
            + "\n2.) Run PointNet++"
            + "\n3.) PointNet++ info"
            + "\n4.) PointNet++ test"
            + "\n5.) Make PointNet++ Dataset"
            + "\n6.) fix pnet"
            + "\n7.) cure experiment"
            + "\n8.) csv fix"
            + "\n9.) csv rename"
            + "\n10.) csv split"
            + "\n11.) make comb ds"
            + "\nSelection: "
        )

        if menu_selection == "1":
            print("Auto Downsample Selected:")
            ds_amt_start = float(input("Downsample start value: "))
            ds_amt_end = float(input("Downsample end value: "))
            ds_amt_inc = float(input("Downsample increment value: "))
            self.pcutils.auto_downsample_data(ds_amt_start, ds_amt_end, ds_amt_inc)
        elif menu_selection == "2":
            #pcd, pcd_all = self.run_pnet('./Data/PNetReady/church_registered_ds_0.350_pnet_ready_wtruth.ply', 0.350)
            self.run_pnet('./Data/PNetReady/church_registered_ds_0.175_pnet_ready_wtruth.ply', 0.175)
            #self.view_pnet(pcd_all)
        elif menu_selection == "3":
            self.view_pnet(file = "./Data/PNet/church_registered_ds_0.175_pnet_all_fix.npy")
        elif menu_selection == "4":
            print("deprecated")
        elif menu_selection == "5":
            self.make_pnet("./Data/church_registered_ds_0.175.npy", True, 0.175)
        elif menu_selection == "6":
            self.fix_pnet("./Data/PNet/church_registered_ds_0.175_pnet_all.npy")
        elif menu_selection == '7':
            self.best_cure("./Data/church_registered_ds_0.300.npy", 100, 50, 0.8)
        elif menu_selection == '8':
            self.fix_csv("./Results/test_ds_0.205_algs_aggl_files_pnet.csv")
        elif menu_selection == '9':
            #self.rename_csv("./Results/Done/test_ds_0.050_algs_birch_files_raw.csv")
            base_dir = "./Results/Done/"
            with os.scandir(base_dir) as entries:
                for entry in entries:
                    if entry.is_file() and entry.path[-4:]=='.csv':
                        print(entry.path)
                        self.rename_csv(entry.path)
        elif menu_selection == '10':
            self.split_csv("./Results/old results/test_cure_0.090_raw.csv")
        elif menu_selection == "11":
            self.make_alg_ds(["./Results/Done/kmeans_cc_0.150_100_750.csv", 
                              "./Results/Done/kmeans_pnet_0.205_100_750.csv",
                              "./Results/Done/kmeans_raw_0.050_100_750.csv"],
                             "./Results/Done/kmeans_all_mixedds_100_750.csv")
            self.make_alg_ds(["./Results/Done/aggl_cc_0.195_100_750.csv", 
                              "./Results/Done/aggl_pnet_0.205_100_750.csv",
                              "./Results/Done/aggl_raw_0.205_100_750.csv"],
                             "./Results/Done/aggl_all_mixedds_100_750.csv")
        # else exits
        
    def make_alg_ds(self, csv_list, out_csv_path):
        df0 = pd.read_csv(csv_list[0], sep=',', header=0, index_col=0)
        df1 = pd.read_csv(csv_list[1], sep=',', header=0, index_col=0)
        df2 = pd.read_csv(csv_list[2], sep=',', header=0, index_col=0)
        
        frames = [df0, df1, df2]
        result = pd.concat(frames, ignore_index=True)
        
        result.to_csv(out_csv_path, sep = ',', header=True, index=True)
        
    def best_cure(self, pcd_loc, clusters, reps_max, comp_max):
        pcd_t = np.load(pcd_loc)
        #raw only
        pcd = pcd_t[:,:4] 
        pcd_type = "raw"
        clustering = Clustering(pcd, pcd_t, pcd_type)
        best_time = 0
        best_rep = 0
        best_comp = 0.0
        results_i, results_j= [], []
        repsrange = range(10, reps_max, 10)
        for i in repsrange:
            exec_time, cure_clusters = clustering.cure_clustering(clusters, i, comp_max, ccore=True, timed = True)
            if best_time == 0:
                best_time = exec_time
                best_rep = i
            if exec_time < best_time:
                best_time = exec_time
                best_rep = i
            print('Exec time:', exec_time, 'for reps:', i)
            clustermetrics = ClusterMetrics(pcd_t[:,4:5],pcd_t[:,4:5],cure_clusters,pcd)
            db_score = clustermetrics.run_metric("db")
            print('DB SCORE:', db_score)
            results_i.append([exec_time, db_score])
        print("best time rep:", best_time, 'best n reps:', best_rep)
        comprange = np.arange(0.2, comp_max, 0.1)
        
        for j in comprange:
            exec_time, cure_clusters = clustering.cure_clustering(clusters, best_rep, j, ccore=True, timed = True) 
            if best_time == 0:
                best_time = exec_time
                best_comp = j
            if exec_time < best_time:
                best_time = exec_time
                best_comp = j
            print('Exec time:', exec_time, 'for comp:', j)
            clustermetrics = ClusterMetrics(pcd_t[:,4:5],pcd_t[:,4:5],cure_clusters,pcd)
            db_score = clustermetrics.run_metric("db")
            print('DB SCORE:', db_score)
            results_j.append([exec_time, db_score])
        print("best time comp:", best_time, 'reps used:', best_rep, 'best comp:', best_comp)
        print('results reps:', results_i)
        print('results comp:', results_j)
        
        bestdb = results_i[0][1]
        bestdbtime = results_i[0][0]
        best_index = repsrange[0]
        index = 0
        for res in results_i:
            if res[1] < bestdb:
                bestdb = res[1]
                bestdbtime=res[0]
                best_index = repsrange[index]
            index += 1
        print("best n reps", best_index, 'with db score of', bestdb, 'and time of',bestdbtime)
        
        bestdb = results_j[0][1]
        bestdbtime = results_j[0][0]
        best_index = comprange[0]
        index = 0
        for res in results_j:
            if res[1] < bestdb:
                bestdb = res[1]
                bestdbtime=res[0]
                best_index = comprange[index]
            index += 1
        print("best comp", best_index, 'with db score of', bestdb, 'and time of',bestdbtime)
                
    def fix_csv(self, csv_path):
        df = pd.read_csv(csv_path, sep=',', header=0, index_col=0)
        print(df.head(2))
        print(df.tail(2))
        #old save and move
        #df.dropna(axis = 'columns', inplace=True)
        # df.drop([1,1])
        # print(df.head(2))
        # print(df.tail(2))
        
        out_path = csv_path[:-4] + "_old.csv"
        df.to_csv(out_path, sep = ',', header=True)
        
        #new index reset
        out_path = csv_path[:-4] + "_temp.csv"
        df.to_csv(out_path, sep = ',', header=True, index=False)
        df = pd.read_csv(out_path, sep=',', header=0, index_col=None)
        df.to_csv(csv_path, sep = ',', header=True, index=True)
        print(df.head(2))
        print(df.tail(2))
        
    def rename_csv(self, csv_path):
        csv = pd.read_csv(csv_path, sep=',', header=0, index_col=0)
        ds_amt = np.unique(csv['down_sample_amount'])
        if len(ds_amt)>1:
            print('multiple downsamples', ds_amt)
            print("figure out issue manually")
            exit(1)
        ds_amt = ds_amt[0]
        algs = np.unique(csv['clustering_algorithm'])
        if len(algs)>1:
            print('multiple algs', algs)
            print("figure out issue manually")
            exit(1)
        alg = algs[0]
        data_sets = np.unique(csv['data_set'])
        if len(data_sets)>1:
            print('multiple data sets', data_sets)
            print("figure out issue manually")
            exit(1)
        data_set = data_sets[0]
        clusters_min = csv['n_clusters'].min()
        clusters_max = csv['n_clusters'].max()
        print(clusters_min,clusters_max,alg,data_set,ds_amt)
        output_path = "./Results/NewNew/"+self.name_csv(alg, clusters_min, clusters_max,data_set, ds_amt)
        print('outpath is ', output_path)
        csv.to_csv(output_path, sep = ',', header=True, index=True)
        
    def name_csv(self,alg,cmin,cmax,dataset,dsamt):
        csv_name = alg + '_' + dataset + '_' + str("%.3f" % dsamt) + '_' + str(cmin) + '_' + str(cmax) + ".csv"
        return csv_name
    
    def split_csv(self,csv_path):
        csv = pd.read_csv(csv_path, sep=',', header=0, index_col=0)
        ds_amt = np.unique(csv['down_sample_amount'])
        if len(ds_amt)>1:
            print('multiple downsamples', ds_amt)
            print("figure out ds issue manually")
            exit(1)
        ds_amt = ds_amt[0]
        algs = np.unique(csv['clustering_algorithm'])
        if len(algs)>1:
            print('multiple algs', algs)
            for alg in algs:
                print('process', alg)
                data = csv[(csv['clustering_algorithm']==alg)]
                data_sets = np.unique(data['data_set'])
                if len(data_sets)>1:
                    print('multiple datasets', data_sets)
                    for data_set in data_sets:
                        print('process', data_set)
                        data = csv[(csv['data_set']==data_set)&(csv['clustering_algorithm']==alg)]
                        clusters_min = data['n_clusters'].min()
                        clusters_max = data['n_clusters'].max()
                        print(clusters_min,clusters_max,alg,data_set,ds_amt)
                        output_path = "./Results/New/"+self.name_csv(alg, clusters_min, clusters_max,data_set, ds_amt)
                        print('outpath is ', output_path)
                        data.to_csv(output_path, sep = ',', header=True, index=True)
                else:   
                    data_set = data_sets[0] 
                    print('process', data_set)  
                    clusters_min = data['n_clusters'].min()
                    clusters_max = data['n_clusters'].max()
                    print(clusters_min,clusters_max,alg,data_set,ds_amt)
                    output_path = "./Results/New/"+self.name_csv(alg, clusters_min, clusters_max,data_set, ds_amt)
                    print('outpath is ', output_path)
                    data.to_csv(output_path, sep = ',', header=True, index=True)
        else:
            alg = algs[0]
            print('process', alg)
            data = csv[(csv['clustering_algorithm']==alg)]
            data_sets = np.unique(data['data_set'])
            if len(data_sets)>1:
                for data_set in data_sets:
                    print('process', data_set)
                    data = csv[(csv['data_set']==data_set)&(csv['clustering_algorithm']==alg)]
                    clusters_min = data['n_clusters'].min()
                    clusters_max = data['n_clusters'].max()
                    print(clusters_min,clusters_max,alg,data_set,ds_amt)
                    output_path = "./Results/New/"+self.name_csv(alg, clusters_min, clusters_max,data_set, ds_amt)
                    print('outpath is ', output_path)
                    data.to_csv(output_path, sep = ',', header=True, index=True)
            else:   
                data_set = data_sets[0]   
                print('process', data_set)
                clusters_min = data['n_clusters'].min()
                clusters_max = data['n_clusters'].max()
                print(clusters_min,clusters_max,alg,data_set,ds_amt)
                output_path = "./Results/New/"+self.name_csv(alg, clusters_min, clusters_max,data_set, ds_amt)
                print('outpath is ', output_path)
                data.to_csv(output_path, sep = ',', header=True, index=True)
        
        
        
    def set_query(self, csv_path):
        df = pd.read_csv(csv_path, sep=',', header=0, index_col=0)
        #print('columns', df.columns)
        #print('indexs', df.index)
        #print(df.keys())
        print(df.head(5))
        print(df.tail(5))
        out_path = csv_path[:-4] + "_temp.csv"
        df.to_csv(out_path, sep = ',', header=True,index=False)
        data = pd.read_csv(out_path, sep=',', header=0, index_col=None)
        print(df.head(5))
        print(df.tail(5))
        algs = data['clustering_algorithm'].head(5)
        print(algs)
        
        arange = df.iloc[42:48]
        print(arange.head(8))
        
        print(data["db"].min())
        print(data[(data["db"]==data["db"].min())])
        


if __name__ == "__main__":
    tools = Tools()
    tools.menu()
