"""
Author: Benny
Date: Nov 2019
"""

# 121: commented out for loop and moved everything back one space 
# 135: feat and xyz list 
# 144: send to classifier - in pointnet2_sem_seg
# 150: add in feat - enter classifier - does forward pass in pointnet2_semseg - get features
# 152: save feat
# 153 and 154: save features and list in a list 
# 163: save np list 

import argparse
import os
from S3DISDataLoader import ScannetDatasetWholeScene
#from data_utils.indoor3d_util import g_label2color
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

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--test_area', type=int, default=5, help='area for testing, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting [default: 5]')
    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    print("experiment dir:", experiment_dir)
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

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

    root = './data'

    TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(root)
    log_string("The number of test data is: %d" % len(TEST_DATASET_WHOLE_SCENE))

    '''MODEL LOADING'''
    print("DIRECTORY:",experiment_dir + '/logs')
    print(os.listdir(experiment_dir + '/logs'))
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')  #log/semseg/pointnet2semseg - and then inside folder - something called checkpoints and then best model inside 
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    #torch_data = torch.Tensor(torch.zeros(2))
    #torch_data = torch_data.float().cuda()
    #numpy_array = torch_data.data.cpu().numpy()
    #print('Before edit:')
    #print(torch_data)
    #print(numpy_array)
    #return 0

    with torch.no_grad():
        scene_id = TEST_DATASET_WHOLE_SCENE.file_list
        scene_id = [x[:-4] for x in scene_id]
        num_batches = len(TEST_DATASET_WHOLE_SCENE)

        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        log_string('---- EVALUATION WHOLE SCENE----')

        for batch_idx in range(num_batches):
            print("Inference [%d/%d] %s ..." % (batch_idx + 1, num_batches, scene_id[batch_idx]))
            #total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
            #total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
            #total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]
            #if args.visual:
            #    fout = open(os.path.join(visual_dir, scene_id[batch_idx] + '_pred.obj'), 'w')
            #    fout_gt = open(os.path.join(visual_dir, scene_id[batch_idx] + '_gt.obj'), 'w')

            whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]    #get all data
            whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]    #get all labels 
            #vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
            #for _ in tqdm(range(args.num_votes), total=args.num_votes):
            #scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx] #dealing with index error
            scene_data = TEST_DATASET_WHOLE_SCENE[batch_idx]
            print("outside")
            num_blocks = scene_data.shape[0]
            s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
            batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))

            #batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
            #batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
            #batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))

            feat_list = torch.tensor([]).cuda()
            xyz_list = torch.tensor([]).cuda()
            #final_feat_list = [] #maybe do empty
            #final_xyz_list = []
            for sbatch in range(s_batch_num):
                #print("sbatch:", sbatch)
                start_idx = sbatch * BATCH_SIZE
                end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                real_batch_size = end_idx - start_idx
                batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                #batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                #batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                #batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                batch_data[:, :, 3:6] /= 1.0

                torch_data = torch.Tensor(batch_data)
                torch_data = torch_data.float().cuda()
                torch_data = torch_data.transpose(2, 1)
                seg_pred, feat = classifier(torch_data)  #send to classifier and get an answer back - changed to feat
                #save features in an npy array - tally up all features
                #print('cat here')
                xyz_list = torch.cat((xyz_list, torch_data[:,:3]), 0)
                feat_list = torch.cat((feat_list, feat), 0)
                #xyz_list.append(torch_data[:,:3])
                #feat_list.append(feat)

                #print("feat_list:", feat_list)
                #print("feat_list SIZE:", feat_list.size)
                #print("xyz_list:", xyz_list)
                #print("xyz_list SIZE:", xyz_list.size)

                #print("feat:", feat)
                #print("torch_data[:,:3]:", torch_data[:,:3])
                
                # feat_list_new = feat.contiguous().cpu().data.numpy()
                # xyz_list_new = torch_data[:,:3].data.cpu().numpy()
                # print('-----------SHAPE-------------')
                # B = feat_list.shape[0]
                # N = feat_list.shape[1]
                # print('feat 0 is ', B)
                # print('feat 1 is ', N)
                # print('------------------------')
                # B = xyz_list.shape[0]
                # N = xyz_list.shape[1]
                # print('xyz 0 is ', B)
                # print('xyz 1 is ', N)
                

                # print("++++++++++++++++++++++++++++")
                # print("feat_list_new:", feat_list_new)
                # print("xyz_list_new:", xyz_list_new)

                # final_feat_list = np.hstack((final_feat_list, feat_list_new))
                # final_xyz_list = np.hstack((final_xyz_list, xyz_list_new))
                # # final_feat_list = np.append(final_feat_list, feat_list_new)
                # # final_xyz_list = np.append(final_xyz_list, xyz_list_new)
                
                # print("final_feat_list:", final_feat_list)
                # print("final_feat_list SIZE:", final_feat_list.size)
                # print("final_xyz_list:", final_xyz_list)
                # print("final_xyz_list SIZE:", final_xyz_list.size)
                # print("yolo")
                #return 0

                #batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                #vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                #                               batch_pred_label[0:real_batch_size, ...],
                #                               batch_smpw[0:real_batch_size, ...])

            #pred_label = np.argmax(vote_label_pool, 1)
            #np.save(xyz_list)
            print("YESSS xyz_list:", xyz_list[0])
            print("YESSS feat_list:", feat_list[0])
            print('-----------SHAPE-------------')
            B = feat_list.shape[0]
            N = feat_list.shape[1]
            print('feat 0 is ', B)
            print('feat 1 is ', N)
            print('------------------------')
            B = xyz_list.shape[0]
            N = xyz_list.shape[1]
            print('xyz 0 is ', B)
            print('xyz 1 is ', N)
            
            #final_feat_list = [] #maybe do empty
            #final_xyz_list = []
            
            #x = type(feat_list)
            #print('type of feat list',x)
            print('feat size', feat_list[0].size())
            print('xyz size', xyz_list[0].size())
            
            tfinal_pcd = torch.column_stack((feat_list, xyz_list))
            print('-----------SHAPE-------------')
            B = tfinal_pcd.shape[0]
            N = tfinal_pcd.shape[1]
            print('tpcd 0 is ', B)
            print('tpcd 1 is ', N)
            print('the pcd tensor!!! ',tfinal_pcd)
            
            
            print('-----------SHAPE-------------')
            B = final_feat_list.shape[0]
            N = final_feat_list.shape[1]
            print('final feat 0 is ', B)
            print('final feat 1 is ', N)
            print('------------------------')
            B = final_xyz_list.shape[0]
            N = final_xyz_list.shape[1]
            print('final xyz 0 is ', B)
            print('final xyz 1 is ', N)
            
            final_feat_list = feat_list.cpu().data.numpy()
            final_xyz_list = xyz_list.cpu().data.numpy()
            
            print("yay final_xyz_list:", final_xyz_list[0])
            print("yay final_feat_list:", final_feat_list[0])
            print('-----------SHAPE-------------')
            B = final_feat_list.shape[0]
            N = final_feat_list.shape[1]
            print('final feat 0 is ', B)
            print('final feat 1 is ', N)
            print('------------------------')
            B = final_xyz_list.shape[0]
            N = final_xyz_list.shape[1]
            print('final xyz 0 is ', B)
            print('final xyz 1 is ', N)
            
            tfinal_pcd = torch.hstack((final_xyz_list, final_feat_list))

            finalPCD = np.hstack((final_xyz_list, final_feat_list))
            print("finalPCD[0]:", finalPCD[0])
            print("finalPCD[1]:", finalPCD[1])
            print("finalPCD[2]:", finalPCD[2])
            print("finalPCD[3]:", finalPCD[3])
            print("finalPCD[4]:", finalPCD[4])
            print("*********************************")

            clustering = Clustering(finalPCD, "3")
            clustering.k_means_clustering_faiss(15)



            #for l in range(NUM_CLASSES):
                #total_seen_class_tmp[l] += np.sum((whole_scene_label == l))
                #total_correct_class_tmp[l] += np.sum((pred_label == l) & (whole_scene_label == l))
                #total_iou_deno_class_tmp[l] += np.sum(((pred_label == l) | (whole_scene_label == l)))
                #total_seen_class[l] += total_seen_class_tmp[l]
                #total_correct_class[l] += total_correct_class_tmp[l]
                #total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

            #iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=np.float) + 1e-6)
            ##print(iou_map)
            #arr = np.array(total_seen_class_tmp)
            #tmp_iou = np.mean(iou_map[arr != 0])
            #log_string('Mean IoU of %s: %.4f' % (scene_id[batch_idx], tmp_iou))
            print('----------------------------')

            #filename = os.path.join(visual_dir, scene_id[batch_idx] + '.txt')
           # with open(filename, 'w') as pl_save:
           #     for i in pred_label:
           #         pl_save.write(str(int(i)) + '\n')
            #    pl_save.close()
            #for i in range(whole_scene_label.shape[0]):
              #  color = g_label2color[pred_label[i]]
             # #  color_gt = g_label2color[whole_scene_label[i]]
              #  if args.visual:
             #       fout.write('v %f %f %f %d %d %d\n' % (
              #          whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color[0], color[1],
              #          color[2]))
              #      fout_gt.write(
              # #         'v %f %f %f %d %d %d\n' % (
             #               whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color_gt[0],
             #               color_gt[1], color_gt[2]))
          #  if args.visual:
           #     fout.close()
           #     fout_gt.close()

       # IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6)
       # iou_per_class_str = '------- IoU --------\n'
      #  for l in range(NUM_CLASSES):
       #     iou_per_class_str += 'class %s, IoU: %.3f \n' % (
        #        seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
        #        total_correct_class[l] / float(total_iou_deno_class[l]))
       # log_string(iou_per_class_str)
      #  log_string('eval point avg class IoU: %f' % np.mean(IoU))
       # log_string('eval whole scene point avg class acc: %f' % (
       #     np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
       # log_string('eval whole scene point accuracy: %f' % (
        #        np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))

        print("Done!")


if __name__ == '__main__':
    
    args = parse_args()
    main(args)