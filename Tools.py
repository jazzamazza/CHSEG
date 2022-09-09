# from datetime import date
import importlib
import os
import sys
from PointCloudUtils import PointCloudUtils
import numpy as np


class Tools:
    def __init__(self):
        self.pcutils = PointCloudUtils()

    def run_pnet(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ROOT_DIR = BASE_DIR
        sys.path.append(os.path.join(ROOT_DIR, "PointNet++"))
        pnet = importlib.import_module("test_semseg")
        return pnet.main_semseg()

    def view_pnet(self):
        pnet_cloud = np.load("./Data/church_registered_ds_0.125.npy")
        self.pcutils.get_attributes(pnet_cloud)
        
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
            + "\nSelection: "
        )

        if menu_selection == "1":
            print("Auto Downsample Selected:")
            ds_amt_start = float(input("Downsample start value: "))
            ds_amt_end = float(input("Downsample end value: "))
            ds_amt_inc = float(input("Downsample increment value: "))
            self.pcutils.auto_downsample_data(ds_amt_start, ds_amt_end, ds_amt_inc)
        elif menu_selection == "2":
            self.run_pnet()
        elif menu_selection == "3":
            self.view_pnet()
        elif menu_selection == "4":
            self.pnet_test()
        elif menu_selection == "5":
            self.make_pnet("./Data/church_registered_ds_0.075.npy", True, 0.075)
        # else exits


if __name__ == "__main__":
    tools = Tools()
    tools.menu()
