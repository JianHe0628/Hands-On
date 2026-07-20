import torch
import lzma
import pickle
import lmdb
from pathlib import Path
import argparse
import subprocess
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def compress_to_lzma(data, filename):
    with lzma.open(filename, "wb") as f:
        pickle.dump(data, f)  # Serialize and compress

def decompress_from_lzma(filename):
    with lzma.open(filename, "rb") as f:
        return pickle.load(f)  # Decompress and deserialize

def load_ham_data(hamer_path):

    if not hamer_path.exists():
        raise FileNotFoundError(f"File not found: {hamer_path}")
    vid_tag = hamer_path.name.removesuffix(".lmdb")

    with lmdb.open(str(hamer_path), readonly=True, lock=False, readahead=False, meminit=False) as env:
        with env.begin() as txn:
            details = pickle.loads(txn.get(key=("details").encode("ascii")))
            hamer_features = torch.zeros((details["num_features"], 2*15*3*3 + 2*1*3*3))
            betas = pickle.loads(txn.get(key=("betas").encode("ascii")))
            pred_cam = pickle.loads(txn.get(key=("pred_cam").encode("ascii")))
            for frame in range(0,details["num_features"]):
                global_orient = (pickle.loads(txn.get(key=f"{vid_tag}_GOrient_{frame}".encode("ascii"))))
                hand_pose = (pickle.loads(txn.get(key=f"{vid_tag}_HPose_{frame}".encode("ascii"))))

                if hand_pose is not None and global_orient is not None:
                    hand_pose = torch.tensor(hand_pose, dtype=torch.float32)
                    global_orient = torch.tensor(global_orient, dtype=torch.float32)
                    
                    if int(hand_pose.shape[0]) == 1 and int(global_orient.shape[0]) == 1:
                        hand_pose = torch.cat((hand_pose, hand_pose), dim=0)
                        global_orient = torch.cat((global_orient, global_orient), dim=0)
                    
                    hand_pose = hand_pose.flatten()
                    global_orient = global_orient.flatten()
                    hamer_features[frame] = torch.cat((hand_pose, global_orient), dim=0)
    return hamer_features, betas, pred_cam

# This function is used to simplify the process of getting the correct bounding box
def show_bbox(img, bbox):
    bbox_tensor = torch.tensor(bbox)
    bbox = bbox_tensor.cpu().numpy()[0]
    x1, y1, x2, y2 = bbox
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Matplotlib
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')
    ax.add_patch(rect)
    plt.show()

def print_gpu_usage():
    try:
        # Run the `nvidia-smi` command
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,nounits,noheader'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print("Error fetching GPU usage:", result.stderr)
            return
        
        # Process the output
        usage_lines = result.stdout.strip().split('\n')
        for i, line in enumerate(usage_lines):
            gpu_util, mem_used, mem_total = map(int, line.split(', '))
            print(f"GPU {i}: Utilization: {gpu_util}%, Memory: {mem_used}/{mem_total} MiB")
    except FileNotFoundError:
        print("nvidia-smi command not found. Ensure NVIDIA drivers are installed.")
    
def get_video_fps(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    print(f"Video FPS: {fps}")
    return fps

def convert_tensors_to_lists(d):
    for key, value in d.items():
        if isinstance(value, torch.Tensor):
            d[key] = value.tolist()  # Convert tensor to list
        elif isinstance(value, dict):  # If there is a nested dictionary
            convert_tensors_to_lists(value)
    return d

def convert_lists_to_tensors(d, device='cuda'):
    for key, value in d.items():
        if isinstance(value, list):
            d[key] = torch.tensor(value, dtype=torch.float32)  # Convert list to tensor
            d[key] = d[key].to(device)
        elif isinstance(value, dict):  # If there is a nested dictionary
            convert_lists_to_tensors(value)
    return d

def load_hamer_features(lmdb_path, video_name, generate_vertices=False):
        env = lmdb.open(
            path=lmdb_path,
            readonly=True,
            readahead=False,
            lock=False,
            meminit=False,
        )

        betas, pred_cam = None, None
        global_orient, hand_pose = [], []
        with env.begin(write=False) as txn:
            details = pickle.loads(txn.get(key=("details").encode("ascii")))
            num_features = details['num_features']
            print(num_features)
            for index in range(num_features):
                global_orient.append(pickle.loads(txn.get(key=f"{video_name}_GOrient_{index}".encode("ascii"))))
                hand_pose.append(pickle.loads(txn.get(key=f"{video_name}_HPose_{index}".encode("ascii"))))

            if generate_vertices:
                betas = pickle.loads(txn.get(key=("betas").encode("ascii")))
                pred_cam = pickle.loads(txn.get(key=("pred_cam").encode("ascii")))
            
        return global_orient, hand_pose, betas, pred_cam, details
