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

    root = "/content/drive/MyDrive/Thesis_Testing/PNET/Data"
    DATASET = ScannetDatasetWholeScene(root)

    '''MODEL LOADING'''
    print("DIRECTORY:",experiment_dir + '/logs')
    print(os.listdir(experiment_dir + '/logs'))
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')  #log/semseg/pointnet2semseg - and then inside folder - something called checkpoints and then best model inside 
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    with torch.no_grad():
        scene_id = DATASET.file_list
        scene_id = [x[:-4] for x in scene_id]
        num_batches = len(DATASET)

        log_string('---- EVALUATION WHOLE SCENE----')

        whole_scene_data = DATASET.scene_points_list[0]    #get all data
        whole_scene_label = DATASET.semantic_labels_list[0]    #get all labels 
        scene_data = DATASET[0]
        print("outside")
        num_blocks = scene_data.shape[0]
        s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
        batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))

        feat_list = torch.tensor([]).cuda()
        xyz_list = torch.tensor([]).cuda()
        for sbatch in range(s_batch_num):
            start_idx = sbatch * BATCH_SIZE
            end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
            real_batch_size = end_idx - start_idx
            batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
            #batch_data[:, :, 3:6] /= 1.0

            torch_data = torch.Tensor(batch_data)
            torch_data = torch_data.float().cuda()
            torch_data = torch_data.transpose(2, 1)
            seg_pred, feat = classifier(torch_data)

            xyz_list = torch.cat((xyz_list, torch_data[:,:3]), 0)
            feat_list = torch.cat((feat_list, feat), 0)
        
        print('feat size', feat_list.size())
        print('xyz size', xyz_list.size())

        final_feat_list = feat_list.cpu().data.numpy()
        final_xyz_list = xyz_list.cpu().data.numpy()
        
        finalPCD = np.hstack((final_xyz_list, final_feat_list))
        print("finalPCD[0]:", finalPCD[0])

        finalPCD_new = finalPCD.reshape(-1,2)
        print("*********************************")

        clustering = Clustering(finalPCD_new, "3")
        clustering.k_means_clustering_faiss(3)

        print("Done!")


if __name__ == '__main__':
    
    args = parse_args()
    main(args)
