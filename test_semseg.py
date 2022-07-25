"""
Author: Benny
Date: Nov 2019
"""

import argparse
import os
from S3DISDataLoader import ScannetDatasetWholeScene
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np
from Clustering import *
import open3d as o3d
from sklearn import preprocessing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    print("experiment dir:", experiment_dir)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = 13
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    DATASET = ScannetDatasetWholeScene()            

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')  #log/semseg/pointnet2semseg - and then inside folder - something called checkpoints and then best model inside 
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    with torch.no_grad():
        log_string('---- EVALUATION WHOLE SCENE----')

        whole_scene_data = DATASET.scene_points_list[0]    #get all data
        whole_scene_label = DATASET.semantic_labels_list[0]    #get all labels   
        scene_data = DATASET[0]
        print("Outside Data Loader")
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

        final_feat_list1 = new_feat_list.copy(order='C')

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

        np.save('/content/drive/MyDrive/GitHubTest/data/finalPCD.npy', finalPCD)
        print("finalPCD shape:", finalPCD.shape)
        print("*********************************")

        clustering = Clustering(finalPCD, "3")
        clustering.k_means_clustering_faiss(20, "CHSEG_finalPCD!")

        # clustering1 = Clustering(normalised_feat, "3")
        # clustering1.k_means_clustering_faiss(3, "CHSEG_feat")

        # clustering2 = Clustering(final_xyz_list, "3")
        # clustering2.k_means_clustering_faiss(3, "CHSEG_xyz_list")

if __name__ == '__main__':
    
    args = parse_args()
    main(args)