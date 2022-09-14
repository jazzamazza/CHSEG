from Clustering import Clustering
from Classification import Classification
from PointCloudLoader import PointCloudLoader
from PointCloudUtils import PointCloudUtils
import numpy as np
from tkinter import filedialog as fd
from tkinter import Tk


def setup(pnet=False, truth=False):
    print("###### POINT CLOUD SETUP ######")

    vis_choice = input(
        "Visualisation:\n Would you like to visualise the selected Point Cloud? [y/n]\n->"
    )
    if vis_choice == "y":
        vis = True
    else:
        vis = False

    ds_choice = input(
        "Downsampling:\n Would you like to downsample the selected Point Cloud? [y/n]\n->"
    )
    if ds_choice == "y":
        ds = True
        ds_amt = float(input("Downsampling:\n Enter the downsample amount.\n->"))
        print("ds amnt =", ds_amt)
    else:
        ds = False
        ds_amt = float(0)

    gui_choice = input(
        "Select file from GUI:\n Would you like to use a GUI to select the Point Cloud file? [y/n]\n->"
    )
    if gui_choice == "y":
        gui = True
    else:
        gui = False

    if gui:
        print("###### POINT CLOUD LOADER ######")
        root = Tk()
        file_types = (
            ("point cloud files", "*.ply *.npy *.las *.xyz *.pcd"),
            ("all files", "*.*"),
        )
        root.filename = fd.askopenfilename(
            title="Select a point cloud file", initialdir="./Data", filetypes=file_types
        )
        root.withdraw()
        file_name = root.filename

        if file_name == "":
            file_path = "./Data/church_registered.npy"
            print("default file selected:", file_path)
        else:
            file_path = file_name
            file_ext = file_name[-4:]
            print("selected file:", file_name)
            print("file ext:", file_ext)
    else:  # HARD CODE PATH
        if input("DS File? (y)") == "y":
            file_path = (
                "./Data/church_registered_ds_"
                + input("ds amnt?: ")
                + input("filetype?: ")
            )
        else:
            file_path = "./Data/church_registered.npy"  # C:\Users\Jared\Code\Thesis\CHSEG\Data\church_registered_alt_dsample_0.05.las
        file_ext = file_path[-4:]
        print("selected file:", file_path)
        print("file ext:", file_ext)

    pc_loader = PointCloudLoader(file_path)

    if pnet:
        # setup point cloud with PointNet++ features
        point_cloud, point_cloud_all = pc_loader.load_point_cloud_pnet(vis, ds, truth = True)
        return point_cloud, point_cloud_all

    if file_ext == ".ply":
        if truth:
            pcd, pcd_truth = pc_loader.load_point_cloud_ply(vis, ds, ds_amt, truth)
            return pcd, pcd_truth
        else:
            pcd = pc_loader.load_point_cloud_ply(vis, ds, ds_amt)
            return pcd

    elif file_ext == ".npy":
        if truth:
            pcd, pcd_all = pc_loader.load_point_cloud_npy(vis, ds, ds_amt, truth)
            return pcd, pcd_all
        else:
            pcd = pc_loader.load_point_cloud_npy(vis, ds, ds_amt)
            return pcd

    elif file_ext == ".las":
        if truth:
            pcd, pcd_all = pc_loader.load_point_cloud_las(vis, ds, ds_amt, truth)
            return pcd, pcd_all
        else:
            pcd = pc_loader.load_point_cloud_las(vis, ds, ds_amt)
            return pcd

    else:
        print("invalid file")
        exit(1)


# interactive application
def application():
    user_input = ""
    while user_input != "q":
        print("--------------Welcome---------------")
        print("Type q to quit the application.")
        print("Choose Loaded Point Cloud Type:")
        user_input = input(
            " 1 : Load Point Cloud"
            + "\n 2 : Load PC + PC w/truth labels"
            + "\n 3 : Load Point Cloud PointNet++"
            + "\n q : or quit the app\n"
            + "\nSelection: "
        )
        if user_input == "q":
            break
        elif user_input == "1":
            point_cloud = setup()
            clustering = Clustering(point_cloud, point_cloud)
        elif user_input == "2":
            point_cloud, pcd_truth = setup(truth=True)
            clustering = Clustering(point_cloud, pcd_truth)
        elif user_input == "3":
            point_cloud, pcd_truth = setup(pnet=True)
            clustering = Clustering(point_cloud, pcd_truth)

        experiment_menu(clustering, user_input)


def experiment_menu(clustering_obj, user_input):
    clustering = clustering_obj
    input_prompt = "\nSet no. clusters: "
    while user_input != "r":
        user_input = input(
            "\nChoose Clustering Method(s):"
            + "\n 1 : K-Means Clustering"
            + "\n 2 : CURE Clustering"
            + "\n 3 : BIRCH Clustering"
            + "\n 4 : Agglomorative Clustering"
            + "\n q : or quit the app"
            + "\n r : Restart the Application\n"
            + "Selection: "
        )
        if user_input == "q":
            exit(0)
        elif user_input == "1":
            clusters = int(input(input_prompt))
            clusters = clustering.k_means_clustering(clusters)
        elif user_input == "2":
            clusters = int(input(input_prompt))
            clustering.cure_clustering(clusters)
        elif user_input == "3":
            clusters = int(input(input_prompt))
            clustering.birch_clustering(clusters)
        elif user_input == "4":
            clusters = int(input(input_prompt))
            clustering.agglomerative_clustering(clusters)


if __name__ == "__main__":
    # unused old main file
    application()
