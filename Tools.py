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

    def menu(self):
        print("Welcome to Tools")
        menu_selection = input(
            "\nPlease select an option from the menu:"
            + "\n1.) Auto Downsample"
            + "\n2.) Run PointNet++"
            + "\n3.) PointNet++ info"
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
        # else exits


if __name__ == "__main__":
    tools = Tools()
    tools.menu()
