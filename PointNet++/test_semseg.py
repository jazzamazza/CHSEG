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
    os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
    NUM_CLASSES = 13
    BATCH_SIZE = 8 
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
        scene_data, scene_label = DATASET[0]
        labels = DATASET.semantic_labels_list
        print("labels", labels)

        print("scene_data", scene_data)
        print("length", len(scene_data[0])) #length is 4096

        print("scene_label", scene_label)
        print("scene_label shape", scene_label.shape) # (799, 4096)

        print("Outside DataLoader")

        num_blocks = scene_data.shape[0]
        s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
        batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))
        
        batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
        
        feat_list, xyz_list, labels_list = [], [], [] 
        saved_first_batch = False
        for sbatch in range(s_batch_num):
            start_idx = sbatch * BATCH_SIZE
            end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
            real_batch_size = end_idx - start_idx
            batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
            
            batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
            
            torch_data = torch.Tensor(batch_data)
            torch_data = torch_data.float().cuda()
            torch_data = torch_data.transpose(2, 1)
            feat, _ = classifier(torch_data)
            
            f = feat.detach().cpu().numpy()
  
            p = torch_data[:,:3,:].transpose(1,2).detach().cpu().numpy()
          


            xyz_list.append(p)
            # feat_list.append(f)
            # labels_list.append(batch_label)
            
            # xyz_list.append(np.copy(batch_data))
            feat_list.append(f)
            labels_list.append(np.copy(batch_label))

            # if not saved_first_batch:
            #   saved_first_batch = True
              # np.save('/content/drive/Shareddrives/CHSEG/LabelTests/testsemseg_batch_loop_hstack.npy', np.hstack((xyz_list, labels_list)))
              # np.save('/content/drive/Shareddrives/CHSEG/LabelTests/testsemseg_batch_loop_column_stack.npy', np.column_stack((xyz_list, labels_list)))

            
        print("labels_list shape", np.shape(labels_list))

        new_feat_list = np.vstack((feat_list))
        new_xyz_list = np.vstack((xyz_list))
        new_labels = np.vstack((labels_list))
        print("new_labels:", new_labels)

        # new_new_labels = np.reshape(new_labels, new_labels.shape + (1,))
        new_new_labels = np.reshape(new_labels, (np.shape(new_labels)[1], -1))
        #new_new_labels = np.reshape(new_labels, (5636096, -1))

        print("new_new_labels", new_new_labels.shape) 
        print('new_feat_list shape:', new_feat_list.shape)
        print('new_xyz_list shape:', new_xyz_list.shape)

        final_feat_list = np.vstack((new_feat_list))
        final_xyz_list = np.vstack((new_xyz_list))
        final_labels = np.vstack((new_new_labels))
        
        print('final_feat_list shape:', final_feat_list.shape)
        print('final_xyz_list shape:', final_xyz_list.shape)
        print('final_labels shape:', final_labels.shape)

        print("final_feat_list:", final_feat_list)
        print("final_xyz_list:", final_xyz_list)
        print("final_labels:", final_labels)

        scalar = preprocessing.MinMaxScaler()
        normalised_feat = scalar.fit_transform(final_feat_list)

        print("Calculating finalPCD")

        finalPCD = np.column_stack((final_xyz_list, normalised_feat))
        finalPCD_all = np.column_stack((final_xyz_list, final_labels, normalised_feat))
        # np.save("/content/drive/Shareddrives/CHSEG/data/pnet_test2_squeeze_vstack.npy", finalPCD_all)

        np.save('./Data/PNet/pnet_final_all', finalPCD_all)
        np.save('./Data/PNet/pnet_final', finalPCD)


        #now we use this .npy file for the rest
        print("finalPCD shape:", finalPCD.shape)
        #Done
        print("*********************************")

        return finalPCD, finalPCD_all

if __name__ == '__main__':
    main_semseg()
