# from datetime import date
import importlib
import os
import sys
from PointCloudUtils import PointCloudUtils
import numpy as np
import pptk

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
        
    def pnet_test(self):
        pcd = np.load("./Data/PNet/church_registered_ds_0.075_0.085_pnet.npy")
        self.pcutils.ds_pnet(pcd, 0.05)
        
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
            self.pnet_test()
        elif menu_selection == "5":
            self.make_pnet("./Data/church_registered_ds_0.175.npy", True, 0.175)
        elif menu_selection == "6":
            self.fix_pnet("./Data/PNet/church_registered_ds_0.175_pnet_all.npy")
        # else exits


if __name__ == "__main__":
    tools = Tools()
    tools.menu()
