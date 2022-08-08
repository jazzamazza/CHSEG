import numpy as np
import open3d as o3d

class DataLoader():
    # prepare to give prediction on each points
    def __init__(self, block_points=4096, stride=0.5, block_size=1.0, padding=0.001):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.stride = stride
        self.scene_points_num = []
        self.scene_points_list = []
        self.semantic_labels_list = []
        
        path = '/content/drive/Shareddrives/CHSEG/data/church_registered_ds_pointnet.ply'
        pcd = o3d.io.read_point_cloud(path)
        ground_truth = np.asarray(pcd.normals)
        print("ground_truth", ground_truth)
        data = np.hstack((np.asarray(pcd.points), np.asarray(pcd.colors), np.asarray(pcd.normals)))
        print("shape", data.shape)
        print(data)

        points = data[:, :3]
        # ground_truth = ground_truth[:,0]
        print(points)
        self.scene_points_list.append(data[:, :6])
        print("=======================")
        print("data[:, :6]", data[:, :6])
        print("=======================")
        # index_ground_truth =  ground_truth[:,0]
        index_ground_truth =  ground_truth[:,0:1]
        self.semantic_labels_list.append(index_ground_truth)
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:,:6]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_room = np.array([])
        label_room = np.array([])
        print("grid_x:", grid_x, ", grid_y:", grid_y)
        for index_y in range(0, grid_y):
            print("index_y:", index_y)
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
                # normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0] #wont look like what you expecting it to
                # normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1] #divides coordinates by maximums - squash into a square 
                # normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                # data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                # data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                label_batch = labels[point_idxs].astype(int)
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)    
                data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch  #normalized - if just take point indexes - also return orginal points - FIRST THREE COLUMNS OF DATA BATCH BEFORE NORMALIZED  
                label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
        data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))
        label_room = label_room.reshape((-1, self.block_points))
        print("data_room shape", data_room.shape)
        print("label_room shape", label_room.shape)
        return data_room, label_room
        # , data[:,6:7]

    def __len__(self):
        return len(self.scene_points_list)
