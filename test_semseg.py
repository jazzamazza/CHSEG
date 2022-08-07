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
from PointCloudLoader import PointCloudLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def main_semseg():
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = '0' #args.gpu

    NUM_CLASSES = 13
    BATCH_SIZE = 16 
    NUM_POINT = 4096 

    #goes through both methods in pointnet data loadeer
    DATASET = DataLoader() 
  
    # print("labels shape", labels.shape())
           

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
        print("labels outside", labels)
        #now has labels appended to it
        print("scene_data", scene_data)
        print("length", len(scene_data[0])) #lenght is 4096

        # print("labels", scene_data[:, :9])
        print("scene_label", scene_label)
        x = scene_label[0]
        print("x", x)
        scene_label_new = np.append(scene_label, x)
        print("scene_label_new shape", scene_label_new.shape)
        print("scene_label_new", scene_label_new)
        print("scene_label shape", scene_label.shape) #(799, 4096)
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
            p = torch_data[:,:3,:].permute(0, 2, 1).detach().cpu().numpy()

            xyz_list.append(p)
            feat_list.append(f)
        
        print("batch label", batch_label)

        new_feat_list = np.vstack((feat_list))
        new_xyz_list = np.vstack((xyz_list))
        new_labels = np.vstack((scene_label_new))

        # new_labels.reshape(3, -1)
        new_new_labels = np.reshape(new_labels, new_labels.shape + (1,))

        print("new_labels", new_new_labels.shape) 
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

        # labels_final = np.vstack((labels))


        print("Calculating finalPCD")

        finalPCD = np.column_stack((final_xyz_list, normalised_feat))
        finalPCD_all = np.column_stack((final_xyz_list, final_labels, normalised_feat))
        # np.save('/content/drive/Shareddrives/CHSEG/data/PNET_TRUTH', finalPCD)
        #finalPCD has points with features - we need to add correct labels to correct points
        
        #now have our dictionary with original points and features
        # dictionary, index_ground_truth = PointCloudLoader.dictionary()
        # print("dict", dictionary)

        # ground_truth_arr = np.asarray([])

        # # final_xyz_list = tuple((final_xyz_list))
        # # print("final xyz tuple", final_xyz_list[0])
        # for i in dictionary:
        #   print("dic", i)
      
        
        # final = np.unique(final_xyz_list)
        # final = tuple(final)
        # print("tup final", tuple(final))
        # for i in range(0, len(final)):
        #   print("==============================")
        #   print("i", i)
        #   # for j in i:
        #   # j = tuple(final[i])
        #   print("final", tuple(final)[i])
        #   print("==============================")
        #     # print("final xyz", final_xyz_list[i][j])
       

        # ground_truth_arr = np.asarray([])
        # for i in final_xyz_list:
        #   j = tuple(i)
        #   # print("================")
        #   # print("i", i)
        #   # print("J", j)
        #   # print("================")
        #   if j in dictionary:
        #     np.append(ground_truth_arr, dictionary[j])
        #     print("in tuple")
        


        # for i in range(0, len(final_xyz_list)):
        #   res = (tuple(final_xyz_list[i])) in dictionary
        #   print("Does tuple exists as dictionary key ? : " + str(res))

          
        # if (tuple(final_xyz_list)) in dict:
        #     np.append(ground_truth_arr, dictionary[final_xyz_list])
            #ground_truth_arr[i] = dict[pnet_tuple]

        # for i in range(0, len(final_xyz_list)):
        #   for key, value in dictionary.items():
        #     if (tuple(final_xyz_list[i])) in key:
        #       dictionary[tuple(final_xyz_list[i])] = value

        #       print("value", value)
          #find the value of the new points in the original dictionary 

        # for key, value in dictionary.items():
        #     dictionary[final_xyz_list[i]] = value

        #now we use this test_semseg_pnet_DS2.npy file for the rest
        print("finalPCD shape:", finalPCD.shape)
        print("*********************************")

        return finalPCD, finalPCD_all
        #clustering = Clustering(finalPCD, "3")
        #clustering.k_means_clustering_faiss(20, "CHSEG_finalPCD!")

if __name__ == '__main__':
    main_semseg()
