import numpy as np
import open3d as o3d
from tqdm import tqdm

class DataLoader():
    # prepare to give prediction on each points
    def __init__(self, path):
        self.block_points = 4096
        self.block_size = 1.0
        self.padding = 0.001
        self.stride = 0.5
        self.scene_points_num, self.scene_points_list, self.semantic_labels_list  = [], [], []

        #path = './Data/PNetReady/church_registered_ds_0.125_pnet_ready_wtruth.ply'
        pcd = o3d.io.read_point_cloud(path)
        #ground_truth = np.ceil(np.asarray(pcd.normals)[:,0:1])
        ground_truth = np.asarray(pcd.normals)[:,0:1]
        for i in range(0, len(ground_truth)):
            # print(truth)
            ground_truth[i][0] = float(round(ground_truth[i][0]))
            if ground_truth[i][0] != float(0) and ground_truth[i][0] != float(1):
                print("error",ground_truth[i][0])
        #print("ground_truth", ground_truth)
        data = np.hstack((np.asarray(pcd.points), np.asarray(pcd.colors))) #xyz intensityx3
        points = data[:, :3]
        print(points)
        self.scene_points_list.append(data[:, :6])
        print("=======================")
        print("data[:, :6]", data[:, :6])
        print("=======================")
        self.semantic_labels_list.append(ground_truth)
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:,:6]  
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_room, label_room = np.array([]), np.array([])
        
        for index_y in tqdm(range(0, grid_y), "loader"):
            #print("index_y:", index_y)
            
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (points[:, 1] >= s_y - self.padding) & (
                                points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.block_points))
                point_size = int(num_batch * self.block_points)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                normlized_xyz = np.zeros((point_size, 3))   #X,Y,Z VALUES CENTERED ON THE ORIGIN
                label_batch = labels[point_idxs].astype(int)
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)                
                data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch  #normalized - if just take point indexes - also return orginal points - FIRST THREE COLUMNS OF DATA BATCH BEFORE NORMALIZED  
                label_room = np.vstack([label_room, label_batch]) if label_room.size else label_batch
                
        data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))
        label_room = label_room.reshape((-1, self.block_points))

        return data_room, label_room

    def __len__(self):
        return len(self.scene_points_list)