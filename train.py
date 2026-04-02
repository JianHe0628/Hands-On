import torch
from pathlib import Path
from utils import decompress_from_lzma

# from models import select_model
from models import select_model
from helpers import (
    load_config,
    find_best_model,
)

torch.set_float32_matmul_precision("medium")
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['TOKENIZERS_PARALLELISM'] = "true"
    
def hamer_inference(cfg_file: str, args):
    print('hamer_inference')
    # Load config
    cfg_file = Path(cfg_file)
    cfg = load_config(cfg_file)
    save_path = Path(cfg["save_path"]) / cfg["name"]

    # Make model
    model = select_model(cfg, model_dir=save_path, return_object=True)

    # Test and Validate model
    model.keep_predictions = True
    model_path = find_best_model(save_path)
    model_path = (save_path / model_path).as_posix()
    sign_prediction_dict = {}

    if (save_path / "last.ckpt").exists() or model_path is not None:
        model = model.load_from_checkpoint(model_path, strict=False)
        model.eval()

        angle_path = Path(args.angle_input)
        hamer_path = Path(args.hamer_input)

        if angle_path.is_dir():
            angle_path_list = angle_path.glob("*.pt")
        elif angle_path.is_file():
            angle_path_list = [angle_path]

        for ap in angle_path_list:
            print(f"Processing {ap.name}")
            file_name = ap.stem
            angle_data = torch.load(ap)
            angles = torch.cat((angle_data['body_angles'], angle_data['right_angles'], angle_data['left_angles']), dim=-1)

            hp = hamer_path / f"{file_name}.lzma" if hamer_path.is_dir() else hamer_path
            hamer_data = decompress_from_lzma(hp)['features']

            # IKSolver drops frames, this is a naive fix
            if 0 < hamer_data.shape[0] - len(angles) < 3:
                pad = hamer_data.shape[0] - len(angles)
                print(f"Padding {pad} frames to {file_name}. Naive Fix, Resolved in FAST Segmentor")
                angles = torch.cat((angles, angles[-1].repeat(pad, 1)), dim=0)
            
            if hamer_data.shape[0] == len(angles):
                sign_prediction, _,__ = model.inference_hamer(cfg=cfg, angels=angles, hamer_features=hamer_data)
                sign_prediction_dict[file_name] = sign_prediction
            else:
                print(f"Angles and hamer features do not match in {file_name}: {hamer_data.shape[0]} != {len(angles)}")

        print(sign_prediction)
        pred_save_path = Path(args.pred_save_path)
        torch.save(sign_prediction_dict, pred_save_path)
        print(f'Segmentation Results stored at {pred_save_path.name}')
    else:
        raise ValueError(f"No model found to test, {save_path}")