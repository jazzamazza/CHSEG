import open3d as o3d
import numpy as np

class PointCloudUtils:

    def downsample_pcd(self, pcd_arr, input_file_format, pnet=True, downsample_amt=0.05):
        # if pnet:
        #     pcd_format = "pnet"
        # else:
        #     pcd_format = "raw"
            
        print("Downsample Called! @", downsample_amt, "on", (input_file_format+" file"), "formating is", format)
        
        if input_file_format==".npy":
            if pnet:
                out_npy, out_ply = self.ds_npy_pnet(pcd_arr, downsample_amt)
        
        out_npy, out_ply = self.ds_npy_raw(pcd_arr, downsample_amt)
        return np.load(out_npy)
        
    def ds_npy(self, pcd_arr, downsample_amt=0.05):
        point_cloud = pcd_arr
        self.get_attributes(point_cloud,"PreDS .npy raw")
        pcloud_shape = np.shape(point_cloud)[0]
            
        # divide pointCloud into points and features 
        points = point_cloud[:,:3]
        intensity = point_cloud[:,3:4] 
        truth_label = point_cloud[:,4:5]

        # format using open3d
        pcd_alt = o3d.geometry.PointCloud()
        pcd_alt.points = o3d.utility.Vector3dVector(points) # add {x,y,z} points to pcd
        rgb = np.hstack((truth_label, np.zeros((pcloud_shape,1)), np.zeros((pcloud_shape,1)))) # form a 3D vector to add to o3d pcd
        normals = np.hstack((intensity, np.zeros((pcloud_shape,1)), np.zeros((pcloud_shape,1)))) # form a 3D vector to add to o3d pcd
        
        pcd_alt.colors = o3d.utility.Vector3dVector(rgb)
        pcd_alt.normals = o3d.utility.Vector3dVector(normals)
        print(pcd_alt)
        
        np_pcloud = np.hstack(((np.asarray(pcd_alt.points)), (np.asarray(pcd_alt.colors)), (np.asarray(pcd_alt.normals)) ))
        npoints = np.shape(np_pcloud)[0]
        # self.get_attributes(np_pcloud, "numpy array original for o3d")
        
        print("*******Downsample start**********")
        downpcd = pcd_alt.voxel_down_sample(voxel_size=downsample_amt)
        print("*******Downsample end**********")
        
        down_np_pcloud = np.hstack(((np.asarray(downpcd.points)), (np.asarray(downpcd.normals)[:,:1]), (np.asarray(downpcd.colors)[:,:1])))
        #self.get_attributes(down_np_pcloud, "downcloud")
        ndownpoints = np.shape(down_np_pcloud)[0]
        self.get_attributes(down_np_pcloud, "numpy array pcloud dsampled")
        
        print("Original Num Points:", npoints, 
              "\nDownsampled Num Points:", ndownpoints, 
              "\nNew is", (100-((ndownpoints/npoints)*100)), "% smaller")
        
        output_path = "./Data/church_registered_dsample_"+str(downsample_amt)
        out_pth_npy = output_path + ".npy"
        out_pth_ply = output_path+".ply"
        print("saving pclouds")
        np.save(out_pth_npy, down_np_pcloud)
        o3d.io.write_point_cloud(out_pth_ply, downpcd)
        return out_pth_npy, out_pth_ply
    
    def ds_ply(self, pcd_03d, downsample_amt=0.05, ):
        point_cloud = pcd_03d
        pcd_points = np.asarry(point_cloud.points)
        pcd_colors = np.asarry(point_cloud.colors)
        pcd_normals = np.asarry(point_cloud.normals)
        pcd_og = np.hstack((pcd_points,pcd_colors,pcd_normals))
        self.get_attributes(pcd_og, "Original PCD")
        down_pcd = point_cloud.voxel_down_sample(voxel_size=downsample_amt)
        
        pcd_points = np.asarry(down_pcd.points)
        pcd_colors = np.asarry(down_pcd.colors)
        pcd_normals = np.asarry(down_pcd.normals)
        pcd_ds = np.hstack((pcd_points,pcd_colors,pcd_normals))
        self.get_attributes(pcd_ds, "Downsample PCD")
        
        #HERE JARED    
        # divide pointCloud into points and features 
        points = down_pcd[:,:3]
        truth_label = down_pcd[:,3:4] 
        intensity = point_cloud[:,6:7]
        
        np_ds = np.hstack((points, intensity, truth_label))
    
        # print("Original Num Points:", npoints, 
        #       "\nDownsampled Num Points:", ndownpoints, 
        #       "\nNew is", (100-((ndownpoints/npoints)*100)), "% smaller")
        
        output_path = "./Data/church_registered_dsample_"+str(downsample_amt)
        out_pth_npy = output_path + ".npy"
        out_pth_ply = output_path+".ply"
        print("saving pclouds")
        np.save(out_pth_npy, np_ds)
        o3d.io.write_point_cloud(out_pth_ply, down_pcd)
        return out_pth_npy, out_pth_ply
    
    def ds_npy_pnet(self, file_path, downsample_amt= 0.05):
        point_cloud = np.load(file_path)
        self.get_attributes(point_cloud,"npy pre pnet")
            
        # divide pointCloud into points and features 
        points = point_cloud[:,:3]
        intensity = point_cloud[:,3:4] 
        truth_label = point_cloud[:,4:5]

        # format using open3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points) # add {x,y,z} points to pcd
        intensity_to_rgb = np.hstack((intensity, intensity, intensity)) # form a 3D vector to add to o3d pcd
        pcd.colors = o3d.utility.Vector3dVector(intensity_to_rgb) # store intensity as every value in color vector
        user_in = input("add truth [y/n]")
        
        if user_in == "y":
            truth = np.hstack((truth_label, truth_label, truth_label))
            pcd.normals = o3d.utility.Vector3dVector(truth)
        print(pcd)
        
        # pnetpcloud = np.hstack(((np.asarray(pcd.points)), (np.asarray(pcd.colors))))
        # npoints = np.shape(pnetpcloud)[0]
        # self.get_attributes(pnetpcloud, "pnet pcloud")

        downpcd = pcd.voxel_down_sample(voxel_size=downsample_amt)
        
        # downpnetpcloud = np.hstack(((np.asarray(downpcd.points)), (np.asarray(downpcd.colors))))
        # ndownpoints = np.shape(downpnetpcloud)[0]
        # self.get_attributes(downpnetpcloud, "pnet pcloud dsampled")
        
        # print("Original Num Points:", npoints, "\nDownsampled Num Points:", ndownpoints, "\nNew is", (100-((ndownpoints/npoints)*100)), "% smaller")
        if user_in == "y":    
            output_path = "./Data/church_registered_pnet_wtruth_"+str(downsample_amt)
            o3d.io.write_point_cloud(output_path+".ply", downpcd)
        else:
            output_path = "./Data/church_registered_pnet_"+str(downsample_amt)
            o3d.io.write_point_cloud(output_path+".ply", downpcd)
            
        # output_path = "./Data/church_registered_pnet_"+str(downsample_amt)
        # np.save(output_path + ".npy", pnetpcloud)
        # o3d.io.write_point_cloud(output_path+".ply", pcd)
    
    def get_attributes(self, pcd, arr_name="Point Cloud"):
        """Prints attributes of given numpy array to console

        Args:
            pcd (Any): Point Cloud Array
        """
        heading_label = arr_name+" Attributes:"
        heading_label += ('\n') + (len(heading_label)*'*')
        print("\n" + heading_label)
        print("\t- Point cloud n points:", np.shape(pcd)[0])
        print("\t- Point cloud dim:", np.ndim(pcd))  
        print("\t- Point cloud shape:", np.shape(pcd))
        print("\t- Point cloud size:", np.size(pcd))
        print("\t- Point cloud data type:", pcd.dtype,'\n')
        