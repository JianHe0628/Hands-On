from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import tempfile
import time
import gc
import shutil
import pickle

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
from hamer.utils.extraction_utils import compress_to_lzma, get_video_fps, print_gpu_usage, convert_tensors_to_lists, show_bbox

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

from vitpose_model import ViTPoseModel

import json
from typing import Dict, Optional


def main(args, model, renderer, device):  
    initial_start_time = time.time()

    # Load detector
    if str(args.bbox).lower() == 'true':
        print("Loading Body Detector to retrieve Bounding Box")
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        if args.body_detector == 'vitdet':
            from detectron2.config import LazyConfig
            import hamer
            cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
            detectron2_cfg = LazyConfig.load(str(cfg_path))
            detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
            detector = DefaultPredictor_Lazy(detectron2_cfg)
        elif args.body_detector == 'regnety':
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
            detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
            detector       = DefaultPredictor_Lazy(detectron2_cfg)

    os.makedirs(args.out_folder, exist_ok=True)
    
    print(f"Total Time for Initialization: {(time.time()-initial_start_time)} Seconds")
    with tempfile.TemporaryDirectory() as temp_dir:
        start_time = time.time()
        if args.vid != '':
            fps = get_video_fps(args.vid)
            image_folder = Path(temp_dir)
            print(f"Temp directory created at {image_folder}")
            os.system(f"ffmpeg -nostdin -i {args.vid} -vf fps={fps} {image_folder}/Frame%d.png")
            print(f"Total Time for Video to Image: {(time.time()-start_time)} Seconds")
        else:
            image_folder = Path(args.img_folder)
            fps = None
        # Get all demo images ends with .jpg or .png
        img_paths = [img for end in args.file_type for img in image_folder.glob(end)]

        #VideoFile Specific
        img_paths = sorted(img_paths,key = lambda x: int(os.path.basename(x).removesuffix('.png').removesuffix('.jpg').removeprefix('Frame')))

        # #For Phoenix
        # img_paths = sorted(img_paths,key = lambda x: int(os.path.basename(x).removeprefix('images').removesuffix('.png')))  

        img_cv2 = cv2.imread(str(img_paths[0]))
        if str(args.bbox).lower() == 'true':
            # Detect humans in image
            det_out = detector(img_cv2)
            det_instances = det_out['instances']
            print_gpu_usage()
            #Clearing memory
            del detector
            torch.cuda.empty_cache()
            gc.collect()

            valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
            if valid_idx[0]:
                pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
                pred_scores=det_instances.scores[valid_idx].cpu().numpy()
                x1, y1, x2, y2 = pred_bboxes[0]
                pred_bboxes[0] = [x1-x1*0.35, y1-y1*0.1, x2+x1*0.35, y2]
            else:
                print("No humans detected in the image")
                return
            print(f"Using BBOX Detected by Vision Model, {pred_bboxes[0]}")

        else:
            # Hardcoded bbox. This assumes person is in the center and that there is always one person in the image
            if args.custom_bbox == '':
                h, w, _ = img_cv2.shape
                pred_bboxes = np.array([[0, 0, w, h]])
                print(f"Using the Entire Video Frame without any Cropping, {pred_bboxes[0]}")
            else:
                pred_bboxes = np.array([list(map(int, args.custom_bbox.split(',')))])
                print(f"Using Custom set BBOX coordinates, {pred_bboxes[0]}")
                
            # Force confidence to be 0.99 that human is present
            pred_scores = np.array([0.99])
        # keypoint detector
        cpm = ViTPoseModel(device)
        print('Loading ViTPose Model')
        print_gpu_usage()
        
        pred_cam_list, global_orient_list, hand_pose_list, betas_list = [], [], [], []
        # Iterate over all images in folder
        for img_path in img_paths:
            img_cv2 = cv2.imread(str(img_path))
            img = img_cv2.copy()[:, :, ::-1]

            # Detect human keypoints for each person
            vitposes_out = cpm.predict_pose(
                img,
                [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
            )

            bboxes = []
            is_right = []
            # Use hands based on hand keypoint detections
            for vitposes in vitposes_out:
                left_hand_keyp = vitposes['keypoints'][-42:-21]
                right_hand_keyp = vitposes['keypoints'][-21:]

                # Rejecting not confident detections
                keyp = left_hand_keyp
                valid = keyp[:,2] > 0.5
                if sum(valid) > 3:
                    bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                    bboxes.append(bbox)
                    is_right.append(0)
                keyp = right_hand_keyp
                valid = keyp[:,2] > 0.5
                if sum(valid) > 3:
                    bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                    bboxes.append(bbox)
                    is_right.append(1)

            if len(bboxes) == 0:
                pred_cam_list.append(None)
                global_orient_list.append(None)
                hand_pose_list.append(None)
                betas_list.append(None)
                continue

            boxes = np.stack(bboxes)
            right = np.stack(is_right)

            # Run reconstruction on all detected hands
            dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
            
            for batch in dataloader:
                batch = recursive_to(batch, device)
                with torch.no_grad():
                    out = model(batch)

                output = convert_tensors_to_lists(out.copy())
                pred_cam_list.append(output['pred_cam'])
                global_orient_list.append(output['pred_mano_params']['global_orient'])
                hand_pose_list.append(output['pred_mano_params']['hand_pose'])
                betas_list.append(output['pred_mano_params']['betas'])
                focal_length = output['focal_length']
            
        assert len(pred_cam_list) == len(global_orient_list) == len(hand_pose_list) == len(betas_list), f"Length of lists are not equal pred_cam_list: {len(pred_cam_list)}, global_orient_list: {len(global_orient_list)}, hand_pose_list: {len(hand_pose_list)}, betas_list: {len(betas_list)}"
        assert len(pred_cam_list) == len(img_paths), f"Length of output features and input images are not equal. features: {len(pred_cam_list)}, img_paths: {len(img_paths)}"
            
        #Create Temp Out Folder for lmdb
        File_Name = os.path.basename(args.vid).removesuffix(".mp4").removesuffix(".mov") if args.vid != '' else os.path.basename(args.img_folder)
        list_length = len(pred_cam_list)
        print(f"Number of features: {list_length}\n")
        
        if args.output_filetype == 'lzma':
            Temp_Database = Path(temp_dir) / f"{File_Name}.lzma"
            Temp_Database.parent.mkdir(parents=True, exist_ok=True)
            assert len(global_orient_list) == len(hand_pose_list), f"Length of global_orient_list {len(global_orient_list)} and hand_pose_list {len(hand_pose_list)} are not equal"
            
            hamer_features = torch.zeros((list_length, 2*15*3*3 + 2*1*3*3))
            for framecount, (hand_pose, global_orient) in enumerate(zip(hand_pose_list, global_orient_list)):
                if hand_pose is not None and global_orient is not None:
                    hand_pose = torch.tensor(hand_pose, dtype=torch.float32)
                    global_orient = torch.tensor(global_orient, dtype=torch.float32)
                    # Hands are Duplicated if only a single hand is detected
                    if int(hand_pose.shape[0]) == 1 and int(global_orient.shape[0]) == 1:
                        hand_pose = torch.cat((hand_pose, hand_pose), dim=0)
                        global_orient = torch.cat((global_orient, global_orient), dim=0)

                    hand_pose = hand_pose.flatten()
                    global_orient = global_orient.flatten()
                    hamer_features[framecount] = torch.cat((hand_pose, global_orient), dim=0)

            hamer_features = hamer_features.numpy()
            final_hamer_dict = {
                "features": hamer_features,
                "pred_cam": pred_cam_list,
                "beta": betas_list,
                "details": {"num_features": list_length, "fps": fps, "focal_length": focal_length, "Bounding_Box": pred_bboxes}
            }
            compress_to_lzma(final_hamer_dict, Temp_Database)
                  
        else:
            print("Output file type not supported. Please use lzma")

        shutil.move(str(Temp_Database), args.out_folder)
    print(f"Total time taken: {(time.time()-initial_start_time)/60:.2f} minutes")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--vid', type=str, default='', help='Path to video file')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    parser.add_argument('--bbox', type=str, default='True', help= 'If set, use provided bbox from ViT')
    parser.add_argument('--custom_bbox', type=str, default='', help='Custom bbox in the format x1,y1,x2,y2')
    parser.add_argument('--MANO_Output', type=bool, default=False, help= 'If set, generate output images')
    parser.add_argument('--output_filetype', type=str, default='lzma', choices=['lmdb', 'lzma'], help='Output file type')

    args = parser.parse_args()

    print(f"Output folder: {args.out_folder}")
    model, model_cfg = load_hamer(args.checkpoint, load_mesh=False) # False sets model to not produce vertice on inference

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()
    renderer = None
    main(args, model, renderer, device)

