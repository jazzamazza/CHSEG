import numpy as np
import open3d as o3d
import laspy as lp
import sys
import importlib
import os
from PointCloudViewer import PointCloudViewer
from PointCloudUtils import PointCloudUtils
from tkinter import filedialog as fd
from tkinter import Tk


class PointCloudLoader:
    """Point Cloud Loader"""

    def __init__(self, path, file_type=None):
        self.pcd_path = path
        print("PCL FILE PATH IS", self.pcd_path)
        self.filetype = self.pcd_path[-4:]
        self.dataset = "raw"
        self.ds = False
        self.ds_amt = None

        path_info = self.pcd_path[:-4].split("_")
        for info in path_info:
            if info == "ds":
                self.ds = True
            if (self.ds) and (info.find("0.") > -1):
                if info.find("x") > -1:
                    self.ds_amt = info
                else:
                    self.ds_amt = float(info)
            if info == "cc":
                self.dataset = "cc"
            if info == "pnet":
                self.dataset = "pnet"

    def file_info(self):
        return {
            "path": self.pcd_path,
            "filetype": self.filetype,
            "dataset": self.dataset,
            "downsampled": self.ds,
            "dsamt": self.ds_amt,
        }

    def load_point_cloud(
        self, vis=False, downsample=False, ds_size=float(0.0), truth=True
    ):
        if self.filetype == ".npy":
            if self.file_info()["dataset"] == "pnet":
                print("in pnet")
                return self.load_point_cloud_pnet(vis, downsample, ds_size, truth)
            else:
                return self.load_point_cloud_npy(vis, downsample, ds_size, truth)
        elif self.filetype == ".ply":
            return self.load_point_cloud_ply(vis, downsample, ds_size, truth)
        elif self.filetype == ".las":
            return self.load_point_cloud_las(vis, downsample, ds_size, truth)

    def load_point_cloud_npy(
        self, vis=False, downsample=False, ds_size=0.0, truth=False
    ):
        print("\n**** Loading Point Cloud (.npy) ****")
        print("File is:", self.pcd_path)
        if downsample:
            print("Downsampling Active @", ds_size)
            pcutils = PointCloudUtils()
            point_cloud = np.load(self.pcd_path)
            ds_path_npy = pcutils.downsample_pcd(point_cloud, ".npy", False, ds_size)
            self.pcd_path = ds_path_npy
            point_cloud = np.load(self.pcd_path)
            self.filetype = ".npy"
            self.get_attributes(point_cloud, "DS Point Cloud")
        else:
            point_cloud = np.load(self.pcd_path)
            self.filetype = ".npy"
            print("**** Point Cloud Loaded ****")
            # self.get_attributes(point_cloud, "Original Point Cloud")

        # divide point_cloud into points and features
        points = point_cloud[:, :3]
        # print("points [0]", points[0])
        intensity = point_cloud[:, 3:4]
        # print("intensity [0]", intensity[0])
        truth_label = point_cloud[:, 4:5]
        # print("truth label [0]", truth_label[0])

        print("\n**** Creating Final Point Cloud w/o GTruth ****")
        final_pcd = np.hstack((points, intensity))  # without truth label
        print("*** Done ***")
        # self.get_attributes(final_pcd, "Point Cloud w/o GTruth")

        if truth:
            print("\n**** Creating Final Point Cloud w/ GTruth ****")
            final_pcd_wtruth = np.hstack((points, intensity, truth_label))
            print("*** Done ***")
            # self.get_attributes(final_pcd_wtruth, "Point Cloud w/ GTruth")

            if vis:
                pview = PointCloudViewer()
                pview.vis_npy(points, intensity, truth_label)
            return final_pcd, final_pcd_wtruth
        else:
            if vis:
                pview = PointCloudViewer()
                pview.vis_npy(points, intensity, truth_label)
            return final_pcd

    def load_point_cloud_ply(
        self, vis=False, downsample=False, down_size=float(0.0), truth=False, has=False
    ):
        if downsample:
            print("Downsampling Active @", down_size)
            pcutils = PointCloudUtils()
            pcd = o3d.io.read_point_cloud(self.pcd_path, print_progress=True)
            ds_ply_path = pcutils.downsample_pcd(
                pcd, self.filetype, False, down_size, truth
            )
            pcd = o3d.io.read_point_cloud(ds_ply_path, print_progress=True)
        else:
            print("\n**** Loading Point Cloud (.ply) ****")
            print("File is:", self.pcd_path)
            # load point cloud .ply file
            path = self.pcd_path
            pcd = o3d.io.read_point_cloud(path, print_progress=True)
            self.filetype = ".ply"
            print("Point Cloud Loaded", pcd)

        if has:
            has_points = pcd.has_points()
            has_colors = pcd.has_colors()
            has_normals = pcd.has_normals()
            has_covariances = pcd.has_covariances()
            print("pcd has points ->", has_points)
            if has_points:
                print(np.asarray(pcd.points))
                # print(np.asarray(pcd.points)[1789886])
            print("pcd has colours ->", has_colors)
            if has_colors:
                print(np.asarray(pcd.colors))
                # print(np.asarray(pcd.colors)[1789886])
            print("pcd has normals ->", has_normals)
            if has_normals:
                print(np.asarray(pcd.normals))
                # print(np.asarray(pcd.normals)[1789886])
            print("pcd has covariances ->", has_covariances)
            if has_covariances:
                print(np.asarray(pcd.covariances))

        pcd_points = np.asarray(pcd.points)
        pcd_intensity = np.asarray(pcd.normals)[:, 0:1]
        pcd_truth = np.asarray(pcd.colors)[:, 0:1]

        print("\n**** Creating Final Point Cloud w/o GTruth ****")
        final_pcd = np.hstack((pcd_points, pcd_intensity))  # without truth label
        print("*** Done ***")
        # self.get_attributes(final_pcd, "Point Cloud w/o GTruth")

        if truth:
            print("\n**** Creating Final Point Cloud w/ GTruth ****")
            final_pcd_wtruth = np.hstack((pcd_points, pcd_intensity, pcd_truth))
            print("*** Done ***")
            # self.get_attributes(final_pcd_wtruth, "Point Cloud w/ GTruth")

            if vis:
                pview = PointCloudViewer()
                pview.vis_ply(pcd, pcd_points, pcd_intensity, pcd_truth)
            return final_pcd, final_pcd_wtruth
        else:
            if vis:
                pview = PointCloudViewer()
                pview.vis_ply(pcd, pcd_points, pcd_intensity, pcd_truth)
            return final_pcd

    def load_point_cloud_las(
        self, vis, downsample=False, ds_size=float(0.05), truth=True
    ):
        print(
            "\n******************Loading Point Cloud with Cloud Compare Generated Features *******************"
        )
        path = self.pcd_path
        self.filetype = ".las"

        # understand las header data
        with lp.open(path) as pcd_f:
            print("Header:", pcd_f.header)
            point_count = pcd_f.header.point_count
            print("Points:", point_count)
        pcd_f.close()

        print("***READING LAS****")
        pcd = lp.read(path)
        # print('Points from Header:', fh.header.point_count)
        # print("Std features:", list(pcd.point_format.standard_dimension_names))
        # print("Cloud Compare Features:", list(pcd.point_format.extra_dimension_names))
        extra_feat_count = len(list(pcd.point_format.extra_dimension_names))
        print("Extra feat count:", extra_feat_count)
        # points = np.transpose(np.vstack((pcd.x, pcd.y, pcd.z)))
        points = pcd.xyz

        for dim in pcd.point_format.extra_dimension_names:
            feat = np.nan_to_num(np.vstack((pcd[dim])))
            max_feat = float(max(feat))
            min_feat = float(min(feat))
            range_feat = max_feat - min_feat

            if max_feat > 1.0:
                for point in feat:
                    if point[0] != 0.0:
                        point[0] = (point[0] - min_feat) / range_feat
            new_max_feat = float(max(feat))
            assert new_max_feat <= 1.0
            points = np.hstack((points, feat))
        truths = points[:, 4:5]
        intensity = points[:, 3:4]
        extra_features = points[:, 5:]
        points = points[:, :3]

        print("\n**** Creating Final Point Cloud w/o GTruth ****")
        # without truth label
        final_pcd = np.hstack((points, intensity, extra_features))
        print("pnet no truth pcloud shape:", np.shape(final_pcd))
        print("*** Done ***")
        
        if vis:
            print("to do vis")
        if downsample:
            # final_pcd = self.voxel_downsample(points, final_features, 18, ds_size)
            print("to do ds")

        if truth == True:
            print("\n**** Creating Final Point Cloud w/GTruth ****")
            final_pcd_wtruth = np.hstack((points, intensity, truths, extra_features))
            print("pnet with truth pcloud shape:", np.shape(final_pcd))
            print("*** Done ***")
            return final_pcd, final_pcd_wtruth
        else:
            return final_pcd

    def load_point_cloud_pnet(
        self, vis=False, downsample=False, ds_size=0.0, truth=False
    ):
        print("\n**** Loading Point Cloud (.npy) ****")
        print("File is:", self.pcd_path)

        point_cloud = np.load(self.pcd_path)
        print("pnet pcloud shape:", np.shape(point_cloud))
        self.filetype = ".npy"
        print("**** Point Cloud Loaded ****")
        # self.get_attributes(point_cloud, "Original Point Cloud")

        # divide point_cloud into points and features
        points = point_cloud[:, :3]
        # print("points [0]", points[0])
        # intensity = point_cloud[:, 3:4]
        # print("intensity [0]", intensity[0])
        truth_label = point_cloud[:, 3:4]
        # print("truth label [0]", truth_label[0])
        pnet_feats = point_cloud[:, 4:]

        print("\n**** Creating Final Point Cloud w/o GTruth ****")
        final_pcd = np.hstack((points, pnet_feats))  # without truth label
        print("pnet no truth pcloud shape:", np.shape(final_pcd))
        print("*** Done ***")
        # self.get_attributes(final_pcd, "Point Cloud w/o GTruth")

        if truth:
            print("\n**** Creating Final Point Cloud w/ GTruth ****")
            final_pcd_wtruth = np.hstack((points, truth_label, pnet_feats))
            print("pnet truth pcloud shape:", np.shape(final_pcd_wtruth))
            print("*** Done ***")
            # self.get_attributes(final_pcd_wtruth, "Point Cloud w/ GTruth")
            return final_pcd, final_pcd_wtruth
        else:
            return final_pcd

    def loadPointCloud_pNet(self, vis=False, downsample=False, ds_size=float(0.0)):
        if downsample:
            print("Downsampling active @", ds_size)
            pcd = o3d.io.read_point_cloud(self.pcd_path, print_progress=True)
            pcutils = PointCloudUtils()
            pcd_pnet_ds_path = pcutils.downsample_pcd(pcd, self.filetype, True, ds_size)
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            ROOT_DIR = BASE_DIR
            sys.path.append(os.path.join(ROOT_DIR, "PointNet++"))
            pnet = importlib.import_module("test_semseg")
            return pnet.main_semseg(pcd_pnet_ds_path)
        else:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            ROOT_DIR = BASE_DIR
            sys.path.append(os.path.join(ROOT_DIR, "PointNet++"))
            pnet = importlib.import_module("test_semseg")
            return pnet.main_semseg(self.pcd_path)

    def get_attributes(self, pcd, arr_name="Point Cloud"):
        """Prints attributes of given numpy array to console

        Args:
            pcd (Any): Point Cloud Array
        """
        heading_label = arr_name + " Attributes:"
        heading_label += ("\n") + (len(heading_label) * "*")
        print("\n" + heading_label)

        print("\t- Point cloud size:", np.size(pcd))
        print("\t- Point cloud dim:", np.ndim(pcd))
        print("\t- Point cloud shape:", np.shape(pcd))
        print("\t- Point cloud points:", np.shape(pcd)[0])
        print("\t- Point cloud data type:", pcd.dtype, "\n")
