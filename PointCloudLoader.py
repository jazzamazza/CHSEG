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
    def __init__(self, path):
        """Point Cloud Loader Class

        Args:
            path (str): file path to point cloud file
        """
        self.pcd_path = path
        print("Point Cloud path is", self.pcd_path)
        self.filetype = self.pcd_path[-4:]
        # defaults
        self.dataset = "raw"
        self.ds = False
        self.ds_amt = None
        
        # set values from file path
        path_info = self.pcd_path[:-4].split("_")
        for info in path_info:
            if info == "ds":
                self.ds = True
            if (self.ds) and (info.find("0.") > -1):
                # fix for old pnet data
                if info.find("x") > -1:
                    # 0.123x0.321
                    self.ds_amt = info
                else:
                    # normal
                    self.ds_amt = float(info)
            if info == "cc":
                self.dataset = "cc"
            if info == "pnet":
                self.dataset = "pnet"
                
        self.pcutils = PointCloudUtils()
        
    def set_file_info(self):
        # set values from file path
        path_info = self.pcd_path[:-4].split("_")
        for info in path_info:
            if info == "ds":
                self.ds = True
            if (self.ds) and (info.find("0.") > -1):
                # fix for old pnet data
                if info.find("x") > -1:
                    # 0.123x0.321
                    self.ds_amt = info
                else:
                    # normal
                    self.ds_amt = float(info)
            if info == "cc":
                self.dataset = "cc"
            if info == "pnet":
                self.dataset = "pnet"

    def file_info(self):
        """create dictionary of file info

        Returns:
            dict: dictionary of file info
        """
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
        """Load point cloud file (default/auto) no need to define filetype

        Args:
            vis (bool, optional): visualise point cloud. Defaults to False.
            downsample (bool, optional): pcd is downsampled. Defaults to False.
            ds_size (float, optional): downsample amount. Defaults to float(0.0).
            truth (bool, optional): include truth labels. Defaults to True.

        Returns:
            ndarray: pcd
            ndarray, ndarray: pcd, pcd w/truth
        """
        if self.filetype == ".npy":
            if self.dataset == "pnet":
                return self.load_point_cloud_pnet(vis, downsample, ds_size, truth)
            else:
                return self.load_point_cloud_npy(vis, downsample, ds_size, truth)
        elif self.filetype == ".ply":
            return self.load_point_cloud_ply(vis, downsample, ds_size, truth)
        elif self.filetype == ".las":
            return self.load_point_cloud_las(vis, downsample, ds_size, truth)

    def load_point_cloud_npy(
        self, vis=False, downsample=False, ds_size=float(0.0), truth=False
    ):
        """Load raw data from .npy file 

        Args:
            vis (bool, optional): visualise point cloud. Defaults to False.
            downsample (bool, optional): pcd is downsampled. Defaults to False.
            ds_size (float, optional): downsample amount. Defaults to float(0.0).
            truth (bool, optional): include truth labels. Defaults to True.

        Returns:
            ndarray: pcd
            ndarray, ndarray: pcd, pcd w/truth
        """
        print("\n**** Loading Point Cloud (.npy) ****")
        print("File is:", self.pcd_path)
        if downsample:
            # downsample if one choses to do so
            print("Downsampling Active @", ds_size)
            point_cloud = np.load(self.pcd_path)
            ds_path_npy = self.pcutils.downsample_pcd(point_cloud, ".npy", False, ds_size)
            self.pcd_path = ds_path_npy
            self.set_file_info()
            point_cloud = np.load(self.pcd_path)
            self.filetype = ".npy"
            print("**** DS Point Cloud Loaded ****")
            self.pcutils.get_attributes(pcd=point_cloud, arr_name = "DS Point Cloud")
        else:
            # load file
            point_cloud = np.load(self.pcd_path)
            self.filetype = ".npy"
            print("**** Point Cloud Loaded ****")
            
        # divide point_cloud into points and features
        points = point_cloud[:, :3]
        intensity = point_cloud[:, 3:4]
        truth_label = point_cloud[:, 4:5]
        # create point cloud with and without truth labels
        print("\n**** Creating Final Point Cloud w/o GTruth ****")
        final_pcd = np.hstack((points, intensity))  # without truth label
        print("*** Done ***")
        
        if vis:
            # visualise pcloud
            pview = PointCloudViewer()
            pview.vis_npy(points, intensity, truth_label)
            
        if truth:  
            # create pcd with truth
            print("\n**** Creating Final Point Cloud w/ GTruth ****")
            final_pcd_wtruth = np.hstack((points, intensity, truth_label))
            print("*** Done ***")
            return final_pcd, final_pcd_wtruth
        else:
            return final_pcd

    def load_point_cloud_ply(
        self, vis=False, downsample=False, down_size=float(0.0), truth=False, has=False
    ):
        """Load raw data from .ply file 

        Args:
            vis (bool, optional): visualise point cloud. Defaults to False.
            downsample (bool, optional): pcd is downsampled. Defaults to False.
            ds_size (float, optional): downsample amount. Defaults to float(0.0).
            truth (bool, optional): include truth labels. Defaults to True.
            has (bool, optional): Check what data ply file includes and print result. Defaults to False.

        Returns:
            ndarray: pcd
            ndarray, ndarray: pcd, pcd w/truth
        """
        if downsample:
            # downsample if one choses to do so
            print("Downsampling Active @", down_size)
            pcd = o3d.io.read_point_cloud(self.pcd_path, print_progress=True)
            ds_ply_path = self.pcutils.downsample_pcd(pcd, self.filetype, False, down_size, truth)
            pcd = o3d.io.read_point_cloud(ds_ply_path, print_progress=True)
            self.pcd_path = ds_ply_path
            self.set_file_info()
            self.filetype = ".ply"
            print("**** DS Point Cloud Loaded ****")
            self.pcutils.get_attributes(pcd=pcd, arr_name = "DS Point Cloud")
        else:
            print("\n**** Loading Point Cloud (.ply) ****")
            print("File is:", self.pcd_path)
            # load point cloud .ply file
            path = self.pcd_path
            pcd = o3d.io.read_point_cloud(path, print_progress=True)
            self.filetype = ".ply"
            print("Point Cloud Loaded", pcd)

        if has:
            # print information about .ply file
            has_points = pcd.has_points()
            has_colors = pcd.has_colors()
            has_normals = pcd.has_normals()
            has_covariances = pcd.has_covariances()
            print("pcd has points ->", has_points)
            if has_points:
                print(np.asarray(pcd.points))
            print("pcd has colours ->", has_colors)
            if has_colors:
                print(np.asarray(pcd.colors))
            print("pcd has normals ->", has_normals)
            if has_normals:
                print(np.asarray(pcd.normals))
            print("pcd has covariances ->", has_covariances)
            if has_covariances:
                print(np.asarray(pcd.covariances))
       
        # split data into points, intensity, truth labels
        pcd_points = np.asarray(pcd.points)
        pcd_intensity = np.asarray(pcd.normals)[:, 0:1]
        pcd_truth = np.asarray(pcd.colors)[:, 0:1]

        #create pcd without truth labels
        print("\n**** Creating Final Point Cloud w/o GTruth ****")
        final_pcd = np.hstack((pcd_points, pcd_intensity))
        print("*** Done ***")
        
        if vis:
            # visualise
                pview = PointCloudViewer()
                pview.vis_ply(pcd, pcd_points, pcd_intensity, pcd_truth)
                
        if truth:
            #create pcd with truth labels
            print("\n**** Creating Final Point Cloud w/ GTruth ****")
            final_pcd_wtruth = np.hstack((pcd_points, pcd_intensity, pcd_truth))
            print("*** Done ***")
            return final_pcd, final_pcd_wtruth
            
        else:
            return final_pcd

    def load_point_cloud_las(
        self, vis, downsample=False, ds_size=float(0.0), truth=True
    ):
        """Load point cloud with geo feats from .las

        Args:
            vis (bool, optional): visualise point cloud. Defaults to False.
            downsample (bool, optional): pcd is downsampled. Defaults to False.
            ds_size (float, optional): downsample amount. Defaults to float(0.0).
            truth (bool, optional): include truth labels. Defaults to True.

        Returns:
            ndarray: pcd
            ndarray, ndarray: pcd, pcd w/truth
        """
        print(
            "\n*** Loading Point Cloud with Cloud Compare Generated Geo Features ***"
        )
        path = self.pcd_path
        self.filetype = ".las"

        # view las header data
        with lp.open(path) as pcd_f:
            print("Header:", pcd_f.header)
            point_count = pcd_f.header.point_count
            print("Points:", point_count)
        pcd_f.close()

        print("*** Reading .las ***")
        # read file
        pcd = lp.read(path)
        # get feature info
        extra_feat_count = len(list(pcd.point_format.extra_dimension_names))
        print("Extra feat count:", extra_feat_count)
        # get points
        points = pcd.xyz

        # add features to numpy array
        for dim in pcd.point_format.extra_dimension_names:
            feat = np.nan_to_num(np.vstack((pcd[dim])))
            max_feat = float(max(feat))
            min_feat = float(min(feat))
            range_feat = max_feat - min_feat
            # normalise features if needed
            if max_feat > 1.0:
                for point in feat:
                    if point[0] != 0.0:
                        point[0] = (point[0] - min_feat) / range_feat
            new_max_feat = float(max(feat))
            #check normalisation
            assert new_max_feat <= 1.0
            # add feat to numpy array
            points = np.hstack((points, feat))
        
        # separate labels
        truths = points[:, 4:5]
        intensity = points[:, 3:4]
        extra_features = points[:, 5:]
        points = points[:, :3]

        print("\n**** Creating Final Point Cloud w/o GTruth ****")
        # without truth label
        final_pcd = np.hstack((points, intensity, extra_features))
        print("pcd no truth pcloud shape:", np.shape(final_pcd))
        print("*** Done ***")
        
        if vis:
            view = PointCloudViewer()
            view.vis_las(points, truths, intensity, extra_features)
        if downsample:
            print("Downsample manually create feats in CloudCompare and load again")

        if truth == True:
            print("\n**** Creating Final Point Cloud w/GTruth ****")
            final_pcd_wtruth = np.hstack((points, intensity, truths, extra_features))
            print("pcd with truth pcloud shape:", np.shape(final_pcd_wtruth))
            print("*** Done ***")
            return final_pcd, final_pcd_wtruth
        else:
            return final_pcd

    def load_point_cloud_pnet(
        self, vis=False, downsample=False, ds_size=0.0, truth=False
    ):
        """Load point cloud with Pointnet++ features from .npy

        Args:
            vis (bool, optional): visualise point cloud. Defaults to False.
            downsample (bool, optional): pcd is downsampled. Defaults to False.
            ds_size (float, optional): downsample amount. Defaults to float(0.0).
            truth (bool, optional): include truth labels. Defaults to True.

        Returns:
            ndarray: pcd
            ndarray, ndarray: pcd, pcd w/truth
        """
        
        print("\n**** Loading Point Cloud (.npy) ****")
        print("File is:", self.pcd_path)
        # load file
        point_cloud = np.load(self.pcd_path)
        print("pnet pcloud shape:", np.shape(point_cloud))
        self.filetype = ".npy"
        print("**** Point Cloud Loaded ****")

        # divide point_cloud into points and features
        points = point_cloud[:, :3]
        truth_label = point_cloud[:, 3:4]
        pnet_feats = point_cloud[:, 4:]
        
        # without truth label
        print("\n**** Creating Final Point Cloud w/o GTruth ****")
        final_pcd = np.hstack((points, pnet_feats))  
        print("pnet no truth pcloud shape:", np.shape(final_pcd))
        print("*** Done ***")

        if truth:
            print("\n**** Creating Final Point Cloud w/ GTruth ****")
            final_pcd_wtruth = np.hstack((points, truth_label, pnet_feats))
            print("pnet truth pcloud shape:", np.shape(final_pcd_wtruth))
            print("*** Done ***")
            if vis:
                view = PointCloudViewer()
                view.vis_npy_pnet_feat(final_pcd_wtruth)
            return final_pcd, final_pcd_wtruth
        else:
            return final_pcd