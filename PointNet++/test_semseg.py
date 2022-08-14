"""
Code adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git
"""

import os
from PointNet_DataLoader import DataLoader
import torch
import sys
import importlib
import numpy as np
from sklearn import preprocessing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))


def main_semseg(pcd_path="./Data/church_registered_pnet.ply"):
    """HYPER PARAMETER"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # args.gpu

    NUM_CLASSES = 13
    BATCH_SIZE = 16
    NUM_POINT = 4096

    # goes through both methods in pointnet data loadeer
    DATASET = DataLoader(block_size=4, path=pcd_path)

    # print("labels shape", labels.shape())

    """MODEL LOADING"""
    model_name = "pointnet2_sem_seg"
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load("PointNet++/best_model.pth")
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier = classifier.eval()

    with torch.no_grad():
        scene_data, scene_label = DATASET[0]
        labels = DATASET.semantic_labels_list
        print("labels outside", labels)
        # now has labels appended to it
        print("scene_data", scene_data)
        print("length", len(scene_data[0]))  # lenght is 4096

        # print("labels", scene_data[:, :9])
        print("scene_label", scene_label)

        scene_label_new = scene_label
        for i in range(0, 11):
            print(i)
            x = scene_label[0]
            print("x", x)
            scene_label_new = np.append(scene_label_new, x)

        # x = scene_label[0]
        # print("x", x)
        # scene_label_new = np.append(scene_label, x)
        # for i in range(0,2):
        #   print(i)
        #   x = scene_label[0]
        #   print("x", x)
        #   scene_label_new = np.append(scene_label_new, x)
        print("scene_label_new shape", scene_label_new.shape)
        print("scene_label_new", scene_label_new)
        print("scene_label shape", scene_label.shape)  # (799, 4096)
        print("Outside DataLoader")
        num_blocks = scene_data.shape[0]
        s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
        batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))
        batch_label = np.zeros((BATCH_SIZE, NUM_POINT))

        feat_list = []
        xyz_list = []
        for sbatch in range(s_batch_num):
            start_idx = sbatch * BATCH_SIZE
            end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
            real_batch_size = end_idx - start_idx
            batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
            batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
            torch_data = torch.Tensor(batch_data)
            torch_data = torch_data.float().cuda()
            torch_data = torch_data.transpose(2, 1)
            feat, seg_pred = classifier(torch_data)

            f = feat.detach().cpu().numpy()
            p = torch_data[:, :3, :].permute(0, 2, 1).detach().cpu().numpy()

            xyz_list.append(p)
            feat_list.append(f)

        print("batch label", batch_label)

        new_feat_list = np.vstack((feat_list))
        new_xyz_list = np.vstack((xyz_list))
        new_labels = np.vstack((scene_label_new))

        # new_labels.reshape(3, -1)
        new_new_labels = np.reshape(new_labels, new_labels.shape + (1,))

        print("new_labels", new_new_labels.shape)
        print("new_feat_list shape:", new_feat_list.shape)
        print("new_xyz_list shape:", new_xyz_list.shape)

        final_feat_list = np.vstack((new_feat_list))
        final_xyz_list = np.vstack((new_xyz_list))
        final_labels = np.vstack((new_new_labels))

        print("final_feat_list shape:", final_feat_list.shape)
        print("final_xyz_list shape:", final_xyz_list.shape)
        print("final_labels shape:", final_labels.shape)

        print("final_feat_list:", final_feat_list)
        print("final_xyz_list:", final_xyz_list)
        print("final_labels:", final_labels)

        scalar = preprocessing.MinMaxScaler()
        normalised_feat = scalar.fit_transform(final_feat_list)

        # labels_final = np.vstack((labels))

        print("Calculating finalPCD")

        finalPCD = np.column_stack((final_xyz_list, normalised_feat))
        finalPCD_all = np.column_stack((final_xyz_list, final_labels, normalised_feat))

        print("finalPCD shape:", finalPCD.shape)
        print("*********************************")

        np.save("./Data/pnetfinalpcd.npy", finalPCD)
        np.save("./Data/pnetfinalpcdall.npy", finalPCD_all)

        return finalPCD, finalPCD_all


if __name__ == "__main__":
    main_semseg()
