# from datetime import date
from PointCloudUtils import PointCloudUtils
import csv
import datetime
import pandas as pd


class AutoTest:
    def __init__(self):
        self.pcutils = PointCloudUtils()
        if input("Are you loading a downsampled file?: ") == "y":
            self.file_path = (
                "./Data/church_registered_ds_"
                + input("ds amnt?: ")
                + "_"
                + input("file type?: ")
            )
        else:
            print("Default file selected.")
            self.file_path = "./Data/church_registered.npy"
        self.file_ext = self.file_path[-4:]
        print("selected file:", self.file_path)
        print("file ext:", self.file_ext)

    def menu(self):
        print("Welcome to AutoTest")
        menu_selection = input(
            "\nPlease select an option from the menu:"
            + "\n1.) Auto Downsample"
            + "\nSelection: "
        )

        if menu_selection == "1":
            print("Auto Downsample Selected:")
            ds_amt_start = float(input("Downsample start value: "))
            ds_amt_end = float(input("Downsample end value: "))
            ds_amt_inc = float(input("Downsample increment value: "))
            self.pcutils.auto_downsample_data(ds_amt_start, ds_amt_end, ds_amt_inc)

    def test_kmeans(self, cluster_start, cluster_end, ds_amt):

        pass

    def data_wrtiter():

        df = pd.DataFrame()


class Test:
    def __init__(
        self,
        algorithm,
        dataset,
        ds_amt,
        n_points,
        n_clusters,
        n_feats,
        feat_list,
        special_info,
    ) -> None:
        self.algorithm = algorithm
        self.dataset = dataset
        self.date = datetime.date.today()
        self.time = datetime.now().strftime("%H:%M:%S")
        self.ds_amt = float(ds_amt)
        self.n_points = int(n_points)
        self.n_clusters = int(n_clusters)
        self.n_feats = int(n_feats)
        self.feat_list = feat_list
        self.special_info = special_info
        # metrics and their values


if __name__ == "__main__":
    autotest = AutoTest()
    autotest.menu()
