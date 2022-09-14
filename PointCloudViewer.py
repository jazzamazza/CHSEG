import pptk
import open3d as o3d
import numpy as np


class PointCloudViewer:
    """PointCloudViewer for viewing PointClouds"""

    def __init__(self, downsample_o3d=0):
        
        self.downsample_o3d = downsample_o3d

    def vis_npy(self, points, intensity, truth_label):
        options = {0: "O3D", 1: "PPTK"}
        try:
            user_input = int(
                input(
                    "\nVisualisation Menu:\n0 - for Open3D\n1 - for PPTK\nYour selection [0/1]: "
                )
            )

            # Open3D Visualisation
            if options.get(user_input) == "O3D":
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(
                    points
                )  # add {x,y,z} points to pcd
                # build (n,3) vector to store in normals
                zero = truth_label  # placeholder
                raw_features = np.hstack(
                    (intensity, truth_label, zero)
                )  # form a 3D vector to add to o3d pcd
                pcd.normals = o3d.utility.Vector3dVector(
                    raw_features
                )  # store additional features (intensity & truth labels) in pcd.normals
                print(pcd)

                if self.downsample_o3d > 0:
                    downpcd = pcd.voxel_down_sample(
                        voxel_size=self.downsample_o3d
                    )  # downsample pcd
                    o3d.visualization.draw_geometries([downpcd])
                else:
                    o3d.visualization.draw_geometries([pcd], window_name="CHSEG")
                    o3d_vis = o3d.visualization.Visualizer()
                    o3d_vis.add_geometry(pcd)
                    o3d_vis.create_window()

            # PPTK Visualisation
            elif options.get(user_input) == "PPTK":
                print("Visualising in PPTK")
                intensity_1d = intensity.flatten()
                truth_label_1d = truth_label.flatten()
                view = pptk.viewer(points, intensity_1d, truth_label_1d, debug=True)
                view.wait()
                view.close()
                print("PPTK Loaded")

            else:
                print("Invalid option selected")
        except ValueError:
            print("Invalid Input. Please Enter a number.")

    def vis_npy_pnet(self, points, intensity, truth_label):
        options = {0: "O3D", 1: "PPTK"}
        try:
            user_input = int(
                input(
                    "\nVisualisation Menu:\n0 - for Open3D\n1 - for PPTK\nYour selection [0/1]: "
                )
            )

            # Open3D Visualisation
            if options.get(user_input) == "O3D":
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(
                    points
                )  # add {x,y,z} points to pcd
                # build (n,3) vector to store in normals
                zero = truth_label  # placeholder
                raw_features = np.hstack(
                    (intensity, truth_label, zero)
                )  # form a 3D vector to add to o3d pcd
                pcd.normals = o3d.utility.Vector3dVector(
                    raw_features
                )  # store additional features (intensity & truth labels) in pcd.normals
                print(pcd)

                if self.downsample_o3d > 0:
                    downpcd = pcd.voxel_down_sample(
                        voxel_size=self.downsample_o3d
                    )  # downsample pcd
                    o3d.visualization.draw_geometries([downpcd])
                else:
                    o3d.visualization.draw_geometries([pcd], window_name="CHSEG")
                    o3d_vis = o3d.visualization.Visualizer()
                    o3d_vis.add_geometry(pcd)
                    o3d_vis.create_window()

            # PPTK Visualisation
            elif options.get(user_input) == "PPTK":
                print("Visualising in PPTK")
                intensity_1d = intensity.flatten()
                truth_label_1d = truth_label.flatten()
                view = pptk.viewer(points, intensity_1d, truth_label_1d)
                print("PPTK Loaded")

            else:
                print("Invalid option selected")
        except ValueError:
            print("Invalid Input. Please Enter a number.")

    def vis_npy_pnet_feat(self, pcd_all):
        print("Visualising Pnet in PPTK")
        points = pcd_all[:,:3]
        truth = pcd_all[:,3:4]
        pnet_feats = pcd_all[:,4:]
        feats = []
        for i in range(0, 127):
            print("index", i)
            feat_1d = pnet_feats[:, i : (i + 1)]
            feat_1d = feat_1d.flatten()
            feats.append(feat_1d)
            
        view = pptk.viewer(points, debug=True)
        view.set(point_size=0.025)
        view.attributes(*feats)
        view.wait()
        view.close()
    
    def vis_las(self, points, truth, intensity, extra):
        print("Visualising Pnet in PPTK")
        truth1d = truth.flatten()
        intensity1d = intensity.flatten()
        feats = []
        feats.append(truth1d)
        feats.append(intensity1d)
        for i in range(0, (len(extra)-1)):
            print("index", i)
            feat_1d = extra[:, i : (i + 1)]
            feat_1d = feat_1d.flatten()
            feats.append(feat_1d)
        view = pptk.viewer(points, debug=True)
        view.set(point_size=0.025)
        view.attributes(*feats)
        view.wait()
        view.close()
    
    def vis_ply(self, pcd, points, intensity, truth_label):
        options = {0: "O3D", 1: "PPTK"}
        try:
            user_input = int(
                input(
                    "\nVisualisation Menu:\n0 - for Open3D\n1 - for PPTK\nYour selection [0/1]: "
                )
            )

            # Open3D Visualisation
            if options.get(user_input) == "O3D":
                if self.downsample_o3d > 0:
                    downpcd = pcd.voxel_down_sample(
                        voxel_size=self.downsample_o3d
                    )  # downsample pcd
                    o3d.visualization.draw_geometries([downpcd])
                else:
                    if truth_label[0][0] != intensity[0][0]:
                        intensity_to_rgb = np.hstack(
                            (intensity, intensity, intensity)
                        )  # form a 3D vector to add to o3d pcd
                        pcd.colors = o3d.utility.Vector3dVector(
                            intensity_to_rgb
                        )  # store intensity as every value in color vector

                    o3d_vis = o3d.visualization.Visualizer()
                    o3d_vis.create_window(window_name="CHSEG")
                    o3d_vis.add_geometry(pcd)
                    o3d_vis.run()
                    o3d_vis.destroy_window()
                    # PPTK Visualisation
            elif options.get(user_input) == "PPTK":
                print("Visualising in PPTK")
                intensity_1d = intensity.flatten()
                truth_label_1d = truth_label.flatten()
                view = pptk.viewer(points, intensity_1d, truth_label_1d)
                print("PPTK Loaded")

            else:
                print("Invalid option selected")
        except ValueError:
            print("Invalid Input. Please Enter a number.")
            
if __name__ == "__main__":
    pcd_pnet_all = np.load("./Data/PNet/church_registered_ds_0.095_pnet_all_fix.npy")
    pcviewer = PointCloudViewer()
    pcviewer.vis_npy_pnet_feat(pcd_pnet_all)
    
