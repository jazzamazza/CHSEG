import open3d as o3d
import numpy as np


class PointCloudUtils:
    def __init__(self) -> None:
        print("PCU created")

    def auto_downsample_data(
        self,
        ds_amt_start,
        ds_amt_end,
        ds_amt_inc=float(0.025),
        pcd_file="./Data/church_registered.npy",
        input_file_format=".npy",
        pnet=False,
        truth=False,
    ):
        pcd_arr = np.load(pcd_file)
        ds = ds_amt_start
        while ds <= ds_amt_end:
            pcd_out_file = self.downsample_pcd(
                pcd_arr, input_file_format, pnet, ds, truth
            )
            print("DS file saved at", pcd_out_file,"\n")
            ds += ds_amt_inc

    def downsample_pcd(
        self,
        pcd,
        input_file_format,
        pnet=False,
        downsample_amt=float(0.05),
        truth=False,
    ):
        print(
            "Downsample Called! @",
            downsample_amt,
            "on",
            (input_file_format + " file")
        )

        if input_file_format == ".npy":
            if pnet:
                out_npy_path, out_ply_path = self.ds_npy(pcd, downsample_amt)
                out_pnet_path = self.npy_to_pnet(out_npy_path, truth, True, downsample_amt)
                return out_pnet_path
            else:
                out_npy_path, out_ply_path = self.ds_npy(pcd, downsample_amt)
                return out_npy_path

        if input_file_format == ".ply":
            if pnet:
                out_npy_path, out_ply_path = self.ds_ply(pcd, downsample_amt)
                out_pnet_path = self.npy_to_pnet(
                    out_npy_path, truth, True, downsample_amt
                )
                return out_pnet_path
            else:
                out_npy_path, out_ply_path = self.ds_ply(pcd, downsample_amt)
                return out_ply_path

    def ds_npy(self, pcd_arr, downsample_amt=float(0.05)):
        #point_cloud = pcd_arr
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
        # self.get_attributes(down_np_pcloud, "Postdownsampling (.npy raw data)")
        reduction = str("%.3f" % (100 - ((ndownpoints / npoints) * 100))) + "%"
        print("New point cloud is", reduction, "smaller")

        output_path = "./Data/church_registered_ds_" + str("%.3f" % downsample_amt)
        out_pth_npy = output_path + ".npy"
        out_pth_ply = output_path + ".ply"
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
        # self.get_attributes(pcd_ds, "Postdownsampling (.ply raw data)")
        ndownpoints = np.shape(pcd_ds)[0]

        # divide pointCloud into points and features
        points = pcd_ds[:, :3]
        truth_label = pcd_ds[:, 3:4]
        intensity = pcd_ds[:, 6:7]

        np_ds = np.hstack((points, intensity, truth_label))
        reduction = 100 - ((ndownpoints / npoints) * 100)
        print(
            "Orig Num Points:",
            npoints,
            "\nDs Num Points:",
            ndownpoints,
            "\nNew is",
            str("%.3f" % reduction),
            "% smaller",
        )

        output_path = "./Data/church_registered_ds_" + str("%.3f" % downsample_amt)
        out_pth_npy = output_path + ".npy"
        out_pth_ply = output_path + ".ply"
        print("Saving pclouds")
        np.save(out_pth_npy, np_ds)
        print(".npy saved")
        o3d.io.write_point_cloud(out_pth_ply, down_pcd)
        print(".ply saved")

        return out_pth_npy, out_pth_ply

    def npy_to_pnet(self, file_path, is_ds=False, ds_amt=float(0.0)):

        point_cloud = np.load(file_path)
        npoints = np.shape(point_cloud)[1]
        # self.get_attributes(point_cloud, "Orignal PCD")

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
        if is_ds:
            output_path = ("./Data/church_registered_ds_" + str(ds_amt) + "_pnet_ready_wtruth.ply")
            np.save("./Data/church_registered_ds_" + str(ds_amt) + "_pnet_ready_wtruth.npy", pcd_new)
            o3d.io.write_point_cloud(output_path, pcd)
            print("files saved")
        else:
            output_path = "./Data/church_registered_pnet_ready_wtruth.ply"
            np.save("./Data/church_registered_pnet_ready_wtruth.npy", pcd_new)
            o3d.io.write_point_cloud(output_path, pcd)
            print("files saved")

    def ds_pnet(self, pcd, ds_amt):
        points = pcd[:,:3]
        feats = pcd[:,3:]
        feats_len = np.shape(feats)[1]
        print(feats_len)
        
    def voxel_downsample(self, points, features, upperBound, ds_size):
        ds_points = np.array([])
        x, y = 0, 0

        for _ in range(0, upperBound):
            x += 1
            print("===========================i:", y)
            print("points.size:", points.size, "points.shape:", np.shape(points))
            print("features.size:", features.size, "features.shape:", np.shape(features))

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points)
            pc.normals = o3d.utility.Vector3dVector(features[:, y : y + 3])
            pc.colors = o3d.utility.Vector3dVector(features[:, y + 3 : y + 6])

            downpcd = pc.voxel_down_sample(voxel_size=ds_size)
            ds_features = np.hstack(
                (np.asarray(downpcd.normals), np.asarray(downpcd.colors))
            )
            ds_points = np.asarray(downpcd.points)
            print(
                "ds_points.size:",
                ds_points.size,
                "ds_points shape:",
                np.shape(ds_points),
            )
            print(
                "ds_features.size:",
                ds_features.size,
                "ds_features shape:",
                np.shape(ds_features),
            )

            if x == 1:
                total_ds_features = ds_features
            else:
                total_ds_features = np.hstack((total_ds_features, ds_features))
            y = y + 6
            if y >= upperBound:
                break

        finalPCD = np.hstack((ds_points, total_ds_features))
        print("finalPCD.size:", finalPCD.size, "finalPCD.shape():", np.shape(finalPCD))
        print("finalPCD[0]:", finalPCD[0])

        return finalPCD

    def get_attributes(self, pcd, arr_name="Point Cloud"):
        """Prints attributes of given numpy array to console

        Args:
            pcd (Any): Point Cloud Array
        """
        heading_label = "\n" + arr_name + " Attributes:"
        heading_label += "\n" + (len(heading_label) * "*")
        print(heading_label)
        print("\tPoint cloud n points:", np.shape(pcd)[0])
        print("\tPoint cloud dim:", np.ndim(pcd))
        print("\tPoint cloud shape:", np.shape(pcd))
        print("\tPoint cloud size:", np.size(pcd))
        print("\tPoint cloud data type:", pcd.dtype, "\n")
