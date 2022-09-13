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
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def main_semseg(in_path, ds_amt):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
    NUM_CLASSES = 13
    BATCH_SIZE = 4 
    NUM_POINT = 4096 

    DATASET = DataLoader(in_path) 
    # DS_STRING = str("%.3f" % ds_amt)
           
    '''MODEL LOADING'''
    model_name = 'pointnet2_sem_seg'
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    print("Model Loaded")
    checkpoint = torch.load('PointNet++/best_model.pth') 
  
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()
    print("Using PNet++")
    with torch.no_grad():
        print("Loading Data")
        scene_data, scene_label = DATASET[0]
        print("Outside DataLoader")
        # print("scene_data", scene_data)
        # print("length", len(scene_data[0])) #length is 4096

        # print("scene_label", scene_label)
        # print("scene_label shape", scene_label.shape) 

        num_blocks = scene_data.shape[0]
        s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
        batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))
        batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
        feat_list, xyz_list, labels_list = [], [], [] 
        print("Applying PNet++ to Dataset")
        for sbatch in tqdm(range(s_batch_num),"sbatch"):
            start_idx = sbatch * BATCH_SIZE
            end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
            real_batch_size = end_idx - start_idx
            batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]       
            batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
            
            torch_data = torch.Tensor(batch_data).float().cuda().transpose(2, 1)
            feat, _ = classifier(torch_data)
            
            xyz_list.append(torch_data[:,:3,:].transpose(1,2).detach().cpu().numpy())
            feat_list.append(feat.detach().cpu().numpy())
            labels_list.append(np.copy(batch_label))

        # scalar = preprocessing.MinMaxScaler()
        # print("Processing Points")
        # final_xyz = np.vstack((np.vstack((xyz_list))))
        # print("Processing Truths")
        # final_labels = np.vstack((np.reshape(np.vstack((labels_list)), (final_xyz.shape[0], -1))))
        # print("Processing Features")
        # final_features = scalar.fit_transform(np.vstack((np.vstack((feat_list)))))

        # print('final_features shape:', final_features.shape)
        # print('final_xyz shape:', final_xyz.shape)
        # print('final_labels shape:', final_labels.shape)

        # print("final_features:", final_features)
        # print("final_xyz:", final_xyz)
        # print("final_labels:", final_labels)

        #print("Calculating finalPCD")
        #finalPCD = np.column_stack((final_xyz, final_features))
        #print("Saving finalPCD")
        # np.save(('./Data/PNet/church_registered_ds_'+ DS_STRING +'_pnet.npy'), finalPCD)
        # print("Calculating finalPCD w/truth")
        # finalPCD_all = np.column_stack((final_xyz, final_labels, final_features))
        
        # print("Saving finalPCD w/truth")
        # np.save(('./Data/PNet/church_registered_ds_'+ DS_STRING +'_pnet_all.npy'), finalPCD_all)
        # print("*********************************","\nfinalPCD shape:", finalPCD.shape, "\n*********************************")
        # finalPCD, finalPCD_all = createPCD
        return createPCD(xyz_list,labels_list,feat_list, ds_amt)
    
def createPCD(xyz_list, labels_list, feat_list, ds_amt):
    DS_STRING = str("%.3f" % ds_amt)
    scalar = preprocessing.MinMaxScaler()
    
    print("Processing Points")
    final_xyz = np.vstack((np.vstack((xyz_list))))
    print("Processing Truths")
    final_labels = np.vstack((np.reshape(np.vstack((labels_list)), (final_xyz.shape[0], -1))))
    print("Processing Features")
    final_features = scalar.fit_transform(np.vstack((np.vstack((feat_list)))))
    
    #fix duplicates
    final_xyz, unique_point_indicies = np.unique(final_xyz, axis = 0, return_index=True)
    final_labels = final_labels[unique_point_indicies]
    final_features = final_features[unique_point_indicies]
    
    print("Calculating finalPCD")
    finalPCD = np.column_stack((final_xyz, final_features))
    print("Saving finalPCD")
    np.save(('./Data/PNet/church_registered_ds_'+ DS_STRING +'_pnet_fix.npy'), finalPCD)
    
    print("Calculating finalPCD w/truth")
    finalPCD_all = np.column_stack((final_xyz, final_labels, final_features))
    print("Saving finalPCD w/truth")
    np.save(('./Data/PNet/church_registered_ds_'+ DS_STRING +'_pnet_all_fix.npy'), finalPCD_all)
    
    print("*********************************","\nfinalPCD shape:", finalPCD.shape, "\n*********************************")

    return finalPCD, finalPCD_all

if __name__ == '__main__':
    main_semseg('./Data/PNetReady/church_registered_ds_0.125_pnet_ready_wtruth.ply', 0.125)
