import open3d as o3d
import numpy as np
from os.path import exists


class PointCloudUtils:
    def __init__(self) -> None:
        print("Point Cloud Utils created")

    def auto_downsample_data(
        self,
        ds_amt_start,
        ds_amt_end,
        ds_amt_inc=float(0.025),
        pcd_file="./Data/church_registered.npy",
        input_file_format=".npy",
        pnet=False
    ):
        """Create a set of downsampled raw data files .ply and .npy

        Args:
            ds_amt_start (float): start ds amount
            ds_amt_end (float): start ds amount
            ds_amt_inc (float, optional): ds step amount e.g. 0.005. Defaults to float(0.025).
            pcd_file (str, optional): file path (.npy file). Defaults to "./Data/church_registered.npy".
            input_file_format (str, optional): file type. Defaults to ".npy".
            pnet (bool, optional): ds and create pnet file. Defaults to False.
        """
        # check file type
        if (pcd_file[-4:] != ".npy"):
            print("must be .npy")
            return None
        # load .npy file
        pcd_arr = np.load(pcd_file)
        #set ds amt
        ds = round(ds_amt_start, 3)
        while ds <= ds_amt_end:
            # downsample file
            pcd_out_file = self.downsample_pcd(pcd_arr, input_file_format, 
                                               pnet, ds)
            print("DS .npy file saved at", pcd_out_file,"\n")
            ds = round((ds + ds_amt_inc), 3)

    def downsample_pcd(
        self,
        pcd,
        input_file_format,
        pnet=False,
        downsample_amt=float(0.05)
    ):
        """Downsample pcd (auto file type) - will downsample file and save .npy and .ply file incl truth labels

        Args:
            pcd (ndarray): pcd data incl truth labels
            input_file_format (str): file type e.g. (.npy)
            pnet (bool, optional): create downsampled Pointnet++ file. Defaults to False.
            downsample_amt (float, optional): downsample amount. Defaults to float(0.05).

        Returns:
            str: file path
        """
        downsample_amt = float(str("%.3f" % downsample_amt))
        print("Downsample Called! @", str("%.3f" % downsample_amt),
              "on",(input_file_format + " file"))

        # .npy downsampling returns npy path
        if input_file_format == ".npy":
            if pnet:
                out_npy_path, out_ply_path = self.ds_npy(pcd, downsample_amt)
                out_pnet_path = self.npy_to_pnet(out_npy_path, True, downsample_amt)
                return out_pnet_path
            else:
                out_npy_path, out_ply_path = self.ds_npy(pcd, downsample_amt)
                return out_npy_path
            
        # .ply downsampling returns .ply path
        if input_file_format == ".ply":
            if pnet:
                out_npy_path, out_ply_path = self.ds_ply(pcd, downsample_amt)
                out_pnet_path = self.npy_to_pnet(
                    out_npy_path, True, downsample_amt
                )
                return out_pnet_path
            else:
                out_npy_path, out_ply_path = self.ds_ply(pcd, downsample_amt)
                return out_ply_path

    def ds_npy(self, pcd_arr, downsample_amt=float(0.05)):
        """Downsample save and return .npy and .ply file

        Args:
            pcd_arr (ndarray): pcd data incl truth
            downsample_amt (float, optional): downsample amount. Defaults to float(0.05).

        Returns:
            str, str: .npy path, .ply path
        """
        # set paths
        output_path = "./Data/church_registered_ds_" + str("%.3f" % downsample_amt)
        out_pth_npy = output_path + ".npy"
        out_pth_ply = output_path + ".ply"
        
        # don't ds if exists
        if (exists(out_pth_npy) and exists(out_pth_ply)):
            print("Exists!")
            return out_pth_npy, out_pth_ply
        
        # begin downsample
        npoints = np.shape(pcd_arr)[0]
        print("Point cloud with,",npoints,"to be downsampled")
        # divide pointCloud into points and features
        points = pcd_arr[:, :3]
        intensity = pcd_arr[:, 3:4]
        truth_label = pcd_arr[:, 4:5]
        zeros = np.zeros((npoints, 1))

        # format using open3d
        pcd = o3d.geometry.PointCloud()
        # add {x,y,z} points to pcd
        pcd.points = o3d.utility.Vector3dVector(points)
        # form 3D vectors to add to o3d pcd
        rgb = np.hstack((truth_label, zeros, zeros))
        normals = np.hstack((intensity, zeros, zeros))
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        
        #ds rounding
        ds = float(str("%.3f" % downsample_amt))
        print("*** downsample start ***","\namt:",ds)
        downpcd = pcd.voxel_down_sample(voxel_size=ds)
        print("*** downsample end ***")

        down_np_pcloud = np.hstack(
                ((np.asarray(downpcd.points)),
                (np.asarray(downpcd.normals)[:, :1]),
                (np.asarray(downpcd.colors)[:, :1])))
        
        ndownpoints = np.shape(down_np_pcloud)[0]
        print("Downsampled point cloud has,",ndownpoints,"original had,",npoints)
        reduction = str("%.3f" % (100 - ((ndownpoints / npoints) * 100))) + "%"
        print("New point cloud is", reduction, "smaller")

        print("Saving point clouds")
        np.save(out_pth_npy, down_np_pcloud)
        print(".npy saved")
        o3d.io.write_point_cloud(out_pth_ply, downpcd)
        print(".ply saved")
        return out_pth_npy, out_pth_ply

    def ds_ply(
        self,
        pcd_o3d,
        downsample_amt=float(0.05),
    ):
        """Downsample save and return .ply and .npy file

        Args:
            pcd_o3d (ndarray): pcd data incl truth
            downsample_amt (float, optional): downsample amount. Defaults to float(0.05).

        Returns:
            str, str: .npy path, .ply path
        """
        # set paths
        output_path = "./Data/church_registered_ds_" + str("%.3f" % downsample_amt)
        out_pth_npy = output_path + ".npy"
        out_pth_ply = output_path + ".ply"
        if (exists(out_pth_npy) and exists(out_pth_ply)):
            # dont ds if exists
            print("Exists!")
            return out_pth_npy, out_pth_ply
        
        # start downsample
        point_cloud = pcd_o3d
        pcd_points = np.asarray(point_cloud.points)
        npoints = np.shape(pcd_points)[0]
        pcd_colors = np.asarray(point_cloud.colors)
        rgb = np.hstack(
            (pcd_colors[:, :1], np.zeros((npoints, 1)), np.zeros((npoints, 1)))
        )
        pcd_colors = o3d.utility.Vector3dVector(rgb)
        pcd_normals = np.asarray(point_cloud.normals)
        pcd_og = np.hstack((pcd_points, pcd_colors, pcd_normals))
        npoints = np.shape(pcd_og)[0]
        # self.get_attributes(pcd_og, "Predownsampling (.ply raw data)")
        print("*******Downsample start**********")
        down_pcd = point_cloud.voxel_down_sample(
            voxel_size=float(str("%.3f" % downsample_amt))
        )
        print("*******Downsample end**********")

        pcd_points = np.asarray(down_pcd.points)
        pcd_colors = np.asarray(down_pcd.colors)
        pcd_normals = np.asarray(down_pcd.normals)
        pcd_ds = np.hstack((pcd_points, pcd_colors, pcd_normals))
        ndownpoints = np.shape(pcd_ds)[0]

        # divide pointCloud into points and features
        points = pcd_ds[:, :3]
        truth_label = pcd_ds[:, 3:4]
        intensity = pcd_ds[:, 6:7]

        np_ds = np.hstack((points, intensity, truth_label))
        reduction = 100 - ((ndownpoints / npoints) * 100)
        print("Orig Num Points:",npoints,"\nDs Num Points:",
              ndownpoints,"\nNew is",
              str(("%.3f" % reduction)+"%"),"smaller"
        )
        
        # save pcds
        print("Saving pclouds")
        np.save(out_pth_npy, np_ds)
        print(".npy saved")
        o3d.io.write_point_cloud(out_pth_ply, down_pcd)
        print(".ply saved")
        # return paths
        return out_pth_npy, out_pth_ply

    def npy_to_pnet(self, file_path, is_ds=False, ds_amt=float(0.0)):
        """Prepare .npy file for injestion to Pointnet++. Requires x,y,z,r,g,b or x,y,z,i,i,i input.

        Args:
            file_path (str): file path of .npy file
            is_ds (bool, optional): file to prep is downsampled True/False. Defaults to False.
            ds_amt (float, optional): downsample amount. Defaults to float(0.0).
        """
        # load file
        point_cloud = np.load(file_path)
        npoints = np.shape(point_cloud)[1]

        # divide pointCloud into points and features
        points = point_cloud[:, :3]
        intensity = point_cloud[:, 3:4]
        truth_label = point_cloud[:, 4:5]
        zeros = np.zeros((npoints, 1))

        # format using open3d
        pcd = o3d.geometry.PointCloud()
        # add {x,y,z} points to pcd
        pcd.points = o3d.utility.Vector3dVector(points)
        # form a 3D vector to add to o3d pcd
        intensity_to_rgb = np.hstack((intensity, intensity, intensity))
        # store intensity as every value in color vector
        pcd.colors = o3d.utility.Vector3dVector(intensity_to_rgb)
        truth = np.hstack((truth_label, truth_label, truth_label))
        pcd.normals = o3d.utility.Vector3dVector(truth)
        pcd_new = np.hstack((points, intensity_to_rgb, truth))
        
        # set output file paths
        if is_ds:
            output_path = ("./Data/PNetReady/church_registered_ds_" + str("%.3f" % ds_amt) + "_pnet_ready_wtruth.ply")
            np.save("./Data/PNetReady/church_registered_ds_" + str("%.3f" % ds_amt) + "_pnet_ready_wtruth.npy", pcd_new)
            o3d.io.write_point_cloud(output_path, pcd)
            print("files saved")
        else:
            output_path = "./Data/PNetReady/church_registered_pnet_ready_wtruth.ply"
            np.save("./Data/PNetReady/church_registered_pnet_ready_wtruth.npy", pcd_new)
            o3d.io.write_point_cloud(output_path, pcd)
            print("files saved")

    def get_attributes(self, pcd, arr_name="Point Cloud"):
        """Prints attributes of given numpy array to console.

        Args:
            pcd (ndarray): point cloud data array
            arr_name (str, optional): Heading description. Defaults to "Point Cloud".
        """
        # self explanitory
        heading_label = "\n" + arr_name + " Attributes:"
        heading_label += "\n" + (len(heading_label) * "*")
        print(heading_label)
        print("\tPoint cloud n points:", np.shape(pcd)[0])
        print("\tPoint cloud dim:", np.ndim(pcd))
        print("\tPoint cloud shape:", np.shape(pcd))
        print("\tPoint cloud size:", np.size(pcd))
        print("\tPoint cloud data type:", pcd.dtype, "\n")
