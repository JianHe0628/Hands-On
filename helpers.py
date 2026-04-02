import os
import csv
import yaml
import lzma
import torch
import pickle

from pathlib import Path
from torch import Tensor, nn
from typing import Dict, Union, Any

import h5py
from typing import Dict, List
import numpy as np 

SCHEMA_VERSION = "1.0"  # Version of the schema used for this output
# V1.0 schema for the output
SCHEMA_META = {
    "video_name": str,  # Name of the video file
    "fps": int,  # Frames per second of the video, -1 if image sequence
    "num_frames": int,  # Total number of frames in the video
    "resolution": tuple,  # Resolution of the video in (width, height)
    "focal_length": float,  # Focal length used for the camera model
    "img_size": int,  # Size of the input images used for the model
    "bbox_shape": tuple,  # Shape of the bounding boxes, if applicable
    "rescale_factor": float,  # Factor used to rescale the bounding boxes
    "num_betas": int,  # Number of betas used in the model
    "schema_version": str,  # Version of the schema used for this output
}

SCHEMA_DATA = {
    "right": np.bool8,  # [Nd]: 0 for left hand, 1 for right hand
    "img_idx": np.int32,  # [Nd] Index of the image in the video
    "frame_idx": np.int32,  # [N, 2] Per frame, Index of the first detection and count of detections in that frame
    "cam": np.float32, # [Nd, 3] Camera parameters in (x, y, z) format
    "kpts_2d": np.float32, # [Nd, 21, 2] Keypoints in 2D (x, y) format
    "kpts_3d": np.float32, # [Nd, 21, 3] Keypoints in 3D (x, y, z) format
    "bboxes": np.float32,  # [Nd, 4] Bounding boxes in (x1, y1, x2, y2) format
}

SCHEMA_MANO = {
    "global_orient": np.float32, # [Nd, 1, 3, 3] for global_orient
    "hand_pose": np.float32, # [Nd, 15, 3, 3] for hand_pose
    "betas": np.float32,  # [Nd, 10] for betas
}

def load_h5_output(output_path: str, idx=None) -> Dict:
    with h5py.File(output_path, "r") as hf:
        meta = {}
        for key in SCHEMA_META.keys():
            meta[key] = hf.attrs[key]

        if not meta["schema_version"] == SCHEMA_VERSION:
            print(
                f"Schema version mismatch: expected {SCHEMA_VERSION}, got {meta['schema_version']}. There may be compatibility issues."
            )

        frame_index = hf["frame_idx"][:]
        total_frames = frame_index.shape[0]
        assert (
            total_frames == meta["num_frames"]
        ), f"Total frames in output {total_frames} does not match meta {meta['num_frames']}"

        if idx is None:
            idx = list(range(total_frames))  # Load all frames if idx is None
        elif isinstance(idx, int):
            idx = [idx]  # Convert single index to list

        # Convert FRAME_IDX to DETECTION_IDX
        detection_idx = []
        for i in idx:
            if i == -1:
                # If the index is -1, it means no detections in that frame
                continue
            start_idx, count = frame_index[i]
            detection_idx.extend(range(start_idx, start_idx + count))

        loaded_output = {}
        for key in SCHEMA_DATA.keys():
            if key == "frame_idx":
                loaded_output[key] = hf[key][:]
            else:
                loaded_output[key] = hf[key][detection_idx]

        mano_output = {}
        mano_group = hf["mano"]
        for key in SCHEMA_MANO.keys():
            mano_output[key] = mano_group[key][detection_idx]

    loaded_output["meta"] = meta
    loaded_output["mano"] = mano_output

    return loaded_output

def load_compressed_pickle(out_file):
    with lzma.open(out_file, 'rb') as file:
        raw_data = file.read()
        data = pickle.loads(raw_data)
    return data


def load_pickle_file(data_path: Union[str, Path]) -> dict:
    with open(data_path, "rb") as handle:
        csv_data = pickle.load(handle)
    return csv_data


def save_pickle_file(data: Any, path: Union[str, Path]) -> None:
    with open(path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_csv(file_path, delimiter="|"):
    with open(file_path, "r", newline="") as file:
        csv_reader = csv.reader(file, delimiter=delimiter)
        csv_content = [row for row in csv_reader]
    return csv_content


def save_csv(file_path, data):
    with open(file_path, "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(data)


def load_text_file(file_path):
    """
    Reads the contents of a text file and returns them as a string.
    """
    try:
        with open(file_path, "r") as file:
            content = file.readlines()
        content = [line.strip() for line in content]
        return content
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred - {e}")
        return None


def save_text_file(file_path, content):
    """
    Saves the provided content to a text file.
    """
    try:
        if content[0][-1] != "\n":
            content = [line + "\n" for line in content]
        with open(file_path, "w") as file:
            file.writelines(content)
        # print(f"File '{file_path}' successfully saved.")
    except Exception as e:
        print(f"Error: An unexpected error occurred - {e}")


def freeze_params(module: nn.Module) -> None:
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    ones = torch.ones(size, size, dtype=torch.bool)
    return torch.tril(ones, out=ones).unsqueeze(0)


def save_config(config: dict, save_dir: Path) -> None:
    import copy

    local_config = copy.deepcopy(config)
    # Remove non valid datatypes
    for (k, v) in local_config.items():
        if not isinstance(v, (float, int, str, list, dict, tuple)):
            local_config.update({k: str(v)})

    with open(save_dir, "w", encoding="utf-8") as ymlfile:
        yaml.safe_dump(
            local_config,
            ymlfile,
            indent=4,
            default_flow_style=False,
            sort_keys=False,
            width=1000,
        )


def load_config(path: Union[Path, str] = "configs/default.yaml") -> Dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    if isinstance(path, str):
        path = Path(path)
    with path.open("r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


class ConfigurationError(Exception):
    """Custom exception for misspecifications of configuration"""


def dict_to_string(d, indent=0):
    result = ""
    for key, value in d.items():
        result += " " * indent + str(key) + ": "
        if isinstance(value, dict):
            result += "\n" + dict_to_string(value, indent + 2)
        else:
            result += str(value) + "\n"
    return result


def pad_list_of_tensors(tensors, padding_value=0, dtype=torch.float32):
    """
    Pad the first dimension of a list of tensors to the same maximum length.

    Args:
    - tensors (list): List of PyTorch tensors with the same size along the first dimension.
    - max_length (int): The maximum length to pad the tensors to.
    - padding_value (int): The value to use for padding.

    Returns:
    - padded_tensors (Tensor): A stacked tensor with padded sequences.
    """
    try:
        max_length = max([t.size(0) for t in tensors])
    except:
        return None, None

    # Find the length of each tensor in the list
    lengths = torch.tensor([t.size(0) for t in tensors], dtype=dtype)

    # Pad each tensor to the max length
    padded_tensors = [
        torch.cat(
            [
                tensor.to(dtype),
                torch.full((max_length - len(tensor),), padding_value, dtype=dtype),
            ]
        )
        for tensor in tensors
    ]

    # Stack the padded tensors along the first dimension
    padded_tensors = torch.stack(padded_tensors, dim=0)

    return padded_tensors, lengths


def adjust_mask_size(mask: Tensor, batch_size: int, hyp_len: int) -> Tensor:
    """
    Adjust mask size along dim=1. used for forced decoding (trg prompting).

    :param mask: trg prompt mask in shape (batch_size, hyp_len)
    :param batch_size:
    :param hyp_len:
    """
    if mask is None:
        return None

    if mask.size(1) < hyp_len:
        _mask = mask.new_zeros((batch_size, hyp_len))
        _mask[:, :mask.size(1)] = mask
    elif mask.size(1) > hyp_len:
        _mask = mask[:, :hyp_len]
    else:
        _mask = mask
    assert _mask.size(1) == hyp_len, (_mask.size(), batch_size, hyp_len)
    return _mask


def tile(x: Tensor, count: int, dim=0) -> Tensor:
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.

    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    # yapf: disable
    x = (x.view(batch, -1)
         .transpose(0, 1)
         .repeat(count, 1)
         .transpose(0, 1)
         .contiguous()
         .view(*out_size))
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def alternate_join_list(list_a: list, list_b: list):
    x, y = 0, 0
    output_list = []
    for i in range(len(list_a) + len(list_b)):
        if i % 2 == 0:
            output_list.append(list_a[x])
            x += 1
        else:
            output_list.append(list_b[y])
            y += 1
    return output_list


def find_best_model(dir):
    files = [f for f in os.listdir(dir) if f.endswith('.ckpt') and 'model' in f]
    score = []
    for f in files:
        score.append(float(f.split('=')[-1].split('.')[0]))
    best_model = files[score.index(max(score))]
    return best_model


def divide_list_and_round(number, x):
    quotient, remainder = divmod(number, x)
    result = [quotient + 1 if i < remainder else quotient for i in range(x)]
    return result


def change_config(cfg: dict, overrides: dict) -> dict:
    """
    Change configuration according to overrides.

    """
    from transformer.transfomer_configs import transformer_modes
    name = []
    # change t mode first
    for k, v in overrides.items():
        if k == 't_mode':
            config = transformer_modes[v]
            cfg['model']['encoder'] = config['encoder']
            cfg['model']['decoder'] = config['decoder']

    for k, v in overrides.items():
        if k == 'do':
            cfg['model']['encoder']['dropout'] = v
            cfg['model']['decoder']['dropout'] = v
        elif k == 'desc':
            cfg['extra_description'] = v
        elif k == 'subsample':
            cfg['data']['subsample'] = v
        elif k == 'window_size':
            cfg['data']['window_size'] = v
        elif k == 'stride':
            cfg['data']['stride'] = v
        elif k == "temp_kernel":
            cfg['data']['temp_conv_kernel'] = v
            cfg['model']['encoder']['temp_conv_kernel'] = v
        elif k == 'weight_type':
            cfg['model']['weight_type'] = v
        elif k == 'batch_size':
            cfg['data']['train_batch_size'] = v
            cfg['data']['test_batch_size'] = v
        elif k == 'aug_thres':
            cfg['data']['aug_thres'] = v
        elif k == 'gloss_variant':
            cfg['data']['rm_gloss_variant'] = bool(v)
        elif k == 'gloss_weight':
            cfg['model']['gloss_weight'] = v
        elif k == 'sign_weight':
            cfg['model']['sign_weight'] = v
        elif k == 'sentence_weight':
            cfg['model']['sentence_weight'] = v
        elif k == 'converge_value_sign':
            cfg['model']['converge_value_sign'] = v
        elif k == 'converge_value_sent':
            cfg['model']['converge_value_sent'] = v
        elif k == 'learning_rate':
            cfg['model']['learning_rate'] = v

        name.append(f'{k}-{v}')
    name = '_'.join(name)
    if name != '':
        cfg['name'] = cfg['name'] + '_' + name
        cfg['save_path'] = os.path.join(cfg['save_path'], cfg['name'])
    return cfg


def face_token_to_pose(tokens: list, face_dictionary: Dict):
    output = []
    for sequence in tokens:
        face = []
        sequence = sequence.tolist() if isinstance(sequence, Tensor) else sequence
        for token in sequence:
            face.append(face_dictionary[token])
        output.append(face)
    return output


def make_gt_frame_lables(glosses, durations, prefix='Ground Truth'):
    output_lables = []
    for gloss, duration in zip(glosses, durations):
        labels = []
        gloss = gloss.split()
        for g, d in zip(gloss, duration):
            labels.extend([f'{prefix}\n{g}' for _ in range(int(d))])
        output_lables.append(labels)

    return output_lables


def batch_data(data, window_size, stride):
    # apply stride and window to data
    num_batches = (data.shape[0] - window_size) // stride + 1
    # Calculate the starting indices for each batch
    batch_start_indices = torch.arange(0, num_batches * stride, stride)
    # Use advanced indexing to extract the batches
    data = data[batch_start_indices.unsqueeze(1) + torch.arange(window_size)]
    return data


def remove_variant_number(gloss):
    # remove the gloss variant number
    new_gloss = []
    for g in gloss:
        temp = []
        for l in g:
            if not l.isdigit():
                temp.append(l)
            else:
                break
        new_gloss.append("".join(temp))
    return new_gloss
