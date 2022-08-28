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
            print("DS file saved at", pcd_out_file)
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
            (input_file_format + " file"),
            "formating is",
            format,
        )

        if input_file_format == ".npy":
            if pnet:
                out_npy_path, out_ply_path = self.ds_npy(pcd, downsample_amt)
                out_pnet_path = self.npy_to_pnet(
                    out_npy_path, truth, True, downsample_amt
                )
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
        point_cloud = pcd_arr
        self.get_attributes(point_cloud, "Predownsampling (.npy raw data)")
        npoints = np.shape(point_cloud)[0]

        # divide pointCloud into points and features
        points = point_cloud[:, :3]
        intensity = point_cloud[:, 3:4]
        truth_label = point_cloud[:, 4:5]

        # format using open3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)  # add {x,y,z} points to pcd
        rgb = np.hstack(
            (truth_label, np.zeros((npoints, 1)), np.zeros((npoints, 1)))
        )  # form a 3D vector to add to o3d pcd
        normals = np.hstack(
            (intensity, np.zeros((npoints, 1)), np.zeros((npoints, 1)))
        )  # form a 3D vector to add to o3d pcd

        pcd.colors = o3d.utility.Vector3dVector(rgb)
        pcd.normals = o3d.utility.Vector3dVector(normals)

        print("*******Downsample start**********")
        downpcd = pcd.voxel_down_sample(voxel_size=float(str("%.3f" % downsample_amt)))
        print("*******Downsample end**********")

        down_np_pcloud = np.hstack(
            (
                (np.asarray(downpcd.points)),
                (np.asarray(downpcd.normals)[:, :1]),
                (np.asarray(downpcd.colors)[:, :1]),
            )
        )
        ndownpoints = np.shape(down_np_pcloud)[0]
        self.get_attributes(down_np_pcloud, "Postdownsampling (.npy raw data)")

        reduction = 100 - ((ndownpoints / npoints) * 100)
        print(
            "Orig Num Points:",
            npoints,
            "\nDs Num Points:",
            ndownpoints,
            "\nNew is",
            reduction,
            "% smaller",
        )

        output_path = "./Data/church_registered_ds_" + str("%.3f" % downsample_amt)
        out_pth_npy = output_path + ".npy"
        out_pth_ply = output_path + ".ply"
        print("Saving pclouds")
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
        self.get_attributes(pcd_og, "Predownsampling (.ply raw data)")
        print("*******Downsample start**********")
        down_pcd = point_cloud.voxel_down_sample(
            voxel_size=float(str("%.3f" % downsample_amt))
        )
        print("*******Downsample end**********")

        pcd_points = np.asarray(down_pcd.points)
        pcd_colors = np.asarray(down_pcd.colors)
        pcd_normals = np.asarray(down_pcd.normals)
        pcd_ds = np.hstack((pcd_points, pcd_colors, pcd_normals))
        self.get_attributes(pcd_ds, "Postdownsampling (.ply raw data)")
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
            reduction,
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

    def npy_to_pnet(self, file_path, truth=False, is_ds=False, ds_amt=float(0.0)):

        point_cloud = np.load(file_path)
        self.get_attributes(point_cloud, "Orignal PCD")

        # divide pointCloud into points and features
        points = point_cloud[:, :3]
        intensity = point_cloud[:, 3:4]
        truth_label = point_cloud[:, 4:5]

        # format using open3d
        pcd = o3d.geometry.PointCloud()
        # add {x,y,z} points to pcd
        pcd.points = o3d.utility.Vector3dVector(points)
        # form a 3D vector to add to o3d pcd
        intensity_to_rgb = np.hstack((intensity, intensity, intensity))
        # store intensity as every value in color vector
        pcd.colors = o3d.utility.Vector3dVector(intensity_to_rgb)
        if truth:
            truth = np.hstack((truth_label, truth_label, truth_label))
            pcd.normals = o3d.utility.Vector3dVector(truth)
        if truth:
            if is_ds:
                output_path = (
                    "./Data/church_registered_pnet_wtruth_" + str(ds_amt) + ".ply"
                )
                o3d.io.write_point_cloud(output_path, pcd)
            else:
                output_path = "./Data/church_registered_pnet_wtruth_no_ds" + ".ply"
                o3d.io.write_point_cloud(output_path, pcd)
        else:
            if is_ds:
                output_path = "./Data/church_registered_pnet_" + str(ds_amt) + ".ply"
                o3d.io.write_point_cloud(output_path, pcd)
            else:
                output_path = "./Data/church_registered_pnet_no_ds" + ".ply"
                o3d.io.write_point_cloud(output_path, pcd)

    def voxel_downsample(self, points, features, upperBound, ds_size):
        print("Features shape:", np.shape(features))

        ds_points = np.array([])
        x, y = 0, 0

        for i in range(0, upperBound):
            x += 1
            print("===========================i:", y)
            print("points.size:", points.size, "points.shape:", np.shape(points))

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
                "ds_features.size:",
                ds_features.size,
                "ds_features shape:",
                np.shape(ds_features),
            )
            print(
                "ds_points.size:",
                ds_points.size,
                "ds_points shape:",
                np.shape(ds_points),
            )

            if x == 1:
                old_ds_points = ds_points
                total_ds_features = ds_features
            else:
                print(
                    "ds_point and old_ds_points EQUAL?",
                    np.array_equal(old_ds_points, ds_points),
                )
                old_ds_points = ds_points
                total_ds_features = np.hstack((total_ds_features, ds_features))
                print(
                    "total_ds_features.size:",
                    total_ds_features.size,
                    "total_ds_features shape:",
                    np.shape(total_ds_features),
                )

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
