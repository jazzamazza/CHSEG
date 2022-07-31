import open3d as o3d
import numpy as np

class PointCloudUtils:

    def downsample_pcd(self, file_path ,input_file_format="npy", pnet=True, downsample_amt=0.8):
        print("dpcd")
        rootPath = "./Data/"
        inputPath = rootPath+"church_registered.npy"  #path to point cloud file
        self.downsample_amt = downsample_amt
        self.npy_raw(file_path=inputPath,downsample_amt= self.downsample_amt)
        
        #self.npy_raw(file_path=inputPath)
    
    def npy_raw(self, file_path, downsample_amt=0.05):
        point_cloud = np.load(file_path)
        self.get_attributes(point_cloud,"npy pre raw")
        pcloud_shape = np.shape(point_cloud)[0]
            
        # divide pointCloud into points and features 
        points = point_cloud[:,:3]
        intensity = point_cloud[:,3:4] 
        truth_label = point_cloud[:,4:5]

        # format using open3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points) # add {x,y,z} points to pcd
        intensity_to_rgb = np.hstack((intensity, np.zeros((pcloud_shape,1)), np.zeros((pcloud_shape,1)))) # form a 3D vector to add to o3d pcd
        pcd.colors = o3d.utility.Vector3dVector(intensity_to_rgb) # store intensity as every value in color vector
        truth_to_normal = np.hstack((truth_label, np.zeros((pcloud_shape,1)), np.zeros((pcloud_shape,1))))
        pcd.normals = o3d.utility.Vector3dVector(truth_to_normal) #store keep discard as 
        print(pcd)
        
        pnetpcloud = np.hstack(((np.asarray(pcd.points)), (np.asarray(pcd.colors)), (np.asarray(pcd.normals))))
        npoints = np.shape(pnetpcloud)[0]
        self.get_attributes(pnetpcloud, "raw pcloud for o3d")

        downpcd = pcd.voxel_down_sample(voxel_size=downsample_amt)
        #need to test line 38
        downpnetpcloud = np.hstack(((np.asarray(downpcd.points)), (np.asarray(downpcd.colors)[:,0:1]), (np.asarray(downpcd.normals)[:,0:1])))
        self.get_attributes(downpnetpcloud, "downcloud")
        ndownpoints = np.shape(downpnetpcloud)[0]
        self.get_attributes(downpnetpcloud, "pnet pcloud dsampled")
        
        print("Original Num Points:", npoints, "\nDownsampled Num Points:", ndownpoints, "\nNew is", (100-((ndownpoints/npoints)*100)), "% smaller")
        
        output_path = "./Data/church_registered_raw_"+str(downsample_amt)
        np.save(output_path + ".npy", downpnetpcloud)
        o3d.io.write_point_cloud(output_path+".ply", downpcd)    
        
    def npy_pnet(self, file_path, downsample_amt= 0.05):
        point_cloud = np.load(file_path)
        self.get_attributes(point_cloud,"npy pre pnet")
            
        # divide pointCloud into points and features 
        points = point_cloud[:,:3]
        intensity = point_cloud[:,3:4] 
        #truthLabel = pointCloud[:,4:5] not used

        # format using open3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points) # add {x,y,z} points to pcd
        intensity_to_rgb = np.hstack((intensity, intensity, intensity)) # form a 3D vector to add to o3d pcd
        pcd.colors = o3d.utility.Vector3dVector(intensity_to_rgb) # store intensity as every value in color vector
        print(pcd)
        
        pnetpcloud = np.hstack(((np.asarray(pcd.points)), (np.asarray(pcd.colors))))
        npoints = np.shape(pnetpcloud)[0]
        self.get_attributes(pnetpcloud, "pnet pcloud")

        downpcd = pcd.voxel_down_sample(voxel_size=downsample_amt)
        
        downpnetpcloud = np.hstack(((np.asarray(downpcd.points)), (np.asarray(downpcd.colors))))
        ndownpoints = np.shape(downpnetpcloud)[0]
        self.get_attributes(downpnetpcloud, "pnet pcloud dsampled")
        
        print("Original Num Points:", npoints, "\nDownsampled Num Points:", ndownpoints, "\nNew is", (100-((ndownpoints/npoints)*100)), "% smaller")
        
        output_path = "./Data/church_registered_pnet_"+str(downsample_amt)
        np.save(output_path + ".npy", downpnetpcloud)
        o3d.io.write_point_cloud(output_path+".ply", downpcd)
        
        output_path = "./Data/church_registered_pnet_"+str(downsample_amt)
        np.save(output_path + ".npy", pnetpcloud)
        o3d.io.write_point_cloud(output_path+".ply", pcd)
    
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
        # print("\t- Point cloud[0] n points:", np.shape(pcd[0])[0])
        # print("\t- Point cloud[0] dim:", np.ndim(pcd[0]))  
        # print("\t- Point cloud[0] shape:", np.shape(pcd[0]))
        # print("\t- Point cloud[0] size:", np.size(pcd[0]))
        # print("\t- Point cloud[1] n points:", np.shape(pcd[1])[0])
        # print("\t- Point cloud[1] dim:", np.ndim(pcd[1]))  
        # print("\t- Point cloud[1] shape:", np.shape(pcd[0]))
        # print("\t- Point cloud[1] size:", np.size(pcd[1]))
        # print("\t- Point cloud[2] n points:", np.shape(pcd[2])[0])
        # print("\t- Point cloud[2] dim:", np.ndim(pcd[2]))  
        # print("\t- Point cloud[2] shape:", np.shape(pcd[2]))
        # print("\t- Point cloud[2] size:", np.size(pcd[2]))
        # print("\t- Point cloud[0]:", pcd[0])
        # print("\t- Point cloud[0][0]:", pcd[0][0])
        # print("\t- Point cloud[1]:", pcd[1])
        # print("\t- Point cloud[1][0]:", pcd[1][0])
        # print("\t- Point cloud[2]:", pcd[2])
        # print("\t- Point cloud[2][0]:", pcd[2][0])
        #print("\t- Point cloud data type:", pcd.dtype,'\n')
        