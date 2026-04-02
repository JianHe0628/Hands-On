import os
import argparse

from train import hamer_inference


def main():
    # login to wandb
    os.environ["WANDB_API_KEY"] = '' 
    os.environ["WANDB_DIR"] = "/tmp"
    ap = argparse.ArgumentParser("Segmentation Time!!")

    ap.add_argument(
        "mode", choices=["train", "test", "inference_hamer"], help="train a model or test or translate"
    )

    ap.add_argument(
        "config_path", metavar="config-path", type=str, help="path to YAML config file"
    )

    # inference args for hamer + angles
    ap.add_argument("--hamer_input", type=str, help="path to the extracted hamer .lzma files")
    ap.add_argument("--dataset_split", type=str, help="directory to the extracted dataset split files")
    ap.add_argument("--angle_input", type=str, help="path to the extracted angles .pt files")
    ap.add_argument("--pred_save_path", type=str, help="path to save the predictions")

    args = ap.parse_args()

    if args.mode == "inference_hamer":
        hamer_inference(cfg_file=args.config_path, args=args)
    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()
