from Clustering import Clustering
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
        if (input("DS File? (y)")=="y"):
            file_path="./Data/church_registered_ds_" + input("ds amnt?: ") + input("filetype?: ")
        else:
            file_path = "./Data/church_registered.npy"  # C:\Users\Jared\Code\Thesis\CHSEG\Data\church_registered_alt_dsample_0.05.las
        file_ext = file_path[-4:]
        print("selected file:", file_path)
        print("file ext:", file_ext)

    pc_loader = PointCloudLoader(file_path, file_ext)

    if pnet:
        # setup point cloud with PointNet++ features
        point_cloud, point_cloud_all = pc_loader.loadPointCloud_pNet(vis, ds)
        if ds:
            np.save(
                ("./Data/church_registered_final_pnet_" + str(ds_amt) + ".npy"),
                point_cloud,
            )
            np.save(
                ("./Data/church_registered_final_pnet_all_" + str(ds_amt) + ".npy"),
                point_cloud_all,
            )
        else:
            np.save(("./Data/church_registered_final_pnet.npy"), point_cloud)
            np.save(("./Data/church_registered_final_pnet_all.npy"), point_cloud_all)

        if truth:
            return point_cloud, point_cloud_all
        else:
            return point_cloud

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
        pcd = pc_loader.load_point_cloud_las(vis)

        if truth:
            truth = np.load("./Data/church_registered_alt_dsample_0.05.npy")
            truth = truth[:, 4:5]
            pcd_all = np.hstack((pcd, truth))
            return pcd, pcd_all
        else:
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
        )
        if user_input == "q":
            break
        elif user_input == "1":
            point_cloud = setup()
            clustering = Clustering(point_cloud, point_cloud, "raw")
        elif user_input == "2":
            point_cloud, pcd_truth = setup(truth=True)
            clustering = Clustering(point_cloud, pcd_truth, "raw_wtruth")
        elif user_input == "3":
            point_cloud, pcd_truth = setup(pnet=True)
            clustering = Clustering(point_cloud, pcd_truth, "pnet_wtruth")
    
        experiment_menu(clustering, user_input)
        

def experiment_menu(clustering_obj, user_input):
    clustering = clustering_obj
    while user_input != "r":
            user_input = input(
                "\nChoose Clustering Method(s):"
                + "\n 0 : K-Means Clustering fais"
                + "\n 1 : sill"
                + "\n 2 : Birch"
                + "\n 3 : cure"
                + "\n 4 : aprop"
                + "\n 5 : kmed"
                + "\n 6 : qual"
                + "\n 6 : rock"
                + "\n q : or quit the app"
                + "\n r : Restart the Application\n"
            )
            if user_input == "q":
                break
            elif user_input == "0":
                clustering.k_means_clustering(15)
            elif user_input == "1":
                clustering.silhouette()
            elif user_input == "2":
                clustering.birch(13)
            elif user_input == "3":
                clusters = int(input("n clusters: "))
                clustering.cure_clustering(clusters)
            elif user_input == "4":
                clustering.affinity_progpogation_clustering()
            elif user_input == "5":
                clustering.kMediods_clustering(14)
            elif user_input == "6":
                clustering.find_quality()
            elif user_input == "7":
                clusters = int(input("n clusters: "))
                clustering.rock_clustering(clusters)
                
if __name__ == "__main__":
    application()
