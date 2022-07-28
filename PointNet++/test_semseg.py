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
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def main_semseg():
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = '0' #args.gpu

    NUM_CLASSES = 13
    BATCH_SIZE = 16 
    NUM_POINT = 4096 

    DATASET = DataLoader()            

    '''MODEL LOADING'''
    model_name = 'pointnet2_sem_seg'
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load('PointNet++/best_model.pth') 
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    with torch.no_grad():
        scene_data = DATASET[0]
        print("Outside DataLoader")
        num_blocks = scene_data.shape[0]
        s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
        batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))
    
        feat_list = [] 
        xyz_list = []
        for sbatch in range(s_batch_num):
            start_idx = sbatch * BATCH_SIZE
            end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
            real_batch_size = end_idx - start_idx
            batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
            torch_data = torch.Tensor(batch_data)
            torch_data = torch_data.float().cuda()
            torch_data = torch_data.transpose(2, 1)
            feat, seg_pred = classifier(torch_data)
            
            f = feat.detach().cpu().numpy()
            p = torch_data[:,:3,:].permute(0, 2, 1).detach().cpu().numpy()

            xyz_list.append(p)
            feat_list.append(f)

        new_feat_list = np.vstack((feat_list))
        new_xyz_list = np.vstack((xyz_list))

        print('new_feat_list shape:', new_feat_list.shape)
        print('new_xyz_list shape:', new_xyz_list.shape)

        final_feat_list = np.vstack((new_feat_list))
        final_xyz_list = np.vstack((new_xyz_list))
        
        print('final_feat_list shape:', final_feat_list.shape)
        print('final_xyz_list shape:', final_xyz_list.shape)

        print("final_feat_list:", final_feat_list)
        print("final_xyz_list:", final_xyz_list)

        scalar = preprocessing.MinMaxScaler()
        normalised_feat = scalar.fit_transform(final_feat_list)

        print("Calculating finalPCD")
        finalPCD = np.column_stack((final_xyz_list, normalised_feat))
        
        path = "./Data/church_registered_pnet.npy"
        np.save(path, finalPCD)
        print("finalPCD shape:", finalPCD.shape)
        print("*********************************")

        return finalPCD
        
        #clustering = Clustering(finalPCD, "3")
        #clustering.k_means_clustering_faiss(20, "CHSEG_finalPCD!")

if __name__ == '__main__':
    main_semseg()
