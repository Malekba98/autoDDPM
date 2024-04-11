import torch

image_dir = ["./data/ATLAS/splits_over_1/atlas_train_png.csv"]
label_dir = ["./data/ATLAS/splits_over_1/atlas_train_mask_png.csv"]

mask_dir = ["./data/ATLAS/splits_over_1/atlas_train_brain_mask_png.csv"]

from torch.utils.data import DataLoader
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

# Get the parent directory
parent_dir_path = os.path.dirname(dir_path)

# Add the parent directory to the system path
sys.path.append(parent_dir_path)
from data.loaders.ixi_loader import mask_preprocessing_loader
import shutil

from PIL import Image
import numpy as np
import argparse


device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="this script preprocesses the ATLAS dataset for nnUNet"
    )
    parser.add_argument(
        "--preprocess_train", action="store_true", help="preprocess the training set"
    )
    args = parser.parse_args()
    if args.preprocess_train:
        dataset = mask_preprocessing_loader(
            image_dir,
            label_dir=label_dir,
            mask_dir=mask_dir,
            target_size=(128, 128),
            test=False,
        )

        loader = DataLoader(
            dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True
        )

        for data in loader:
            image = data[0].to(device)
            mask = data[1].to(device)
            idx = data[2]
            filename = data[3]
            mask_filename = data[4]
            # copy file with filename to new directory "./nnunet_data/nnunet_raw/Dataset500_ATLAS/imagesTr"
            # copy file with mask_filename to new directory "./nnunet_data/nnunet_raw/Dataset500_ATLAS/labelsTr"
            # ATLAS_001_0000.png
            index = int(filename[0].split("_")[-1].replace(".png", ""))

            image_dest = f"./nnunet_data/nnunet_raw/Dataset500_ATLAS/imagesTr/ATLAS_{index:03d}_0000.png"
            mask_dest = f"./nnunet_data/nnunet_raw/Dataset500_ATLAS/labelsTr/ATLAS_{index:03d}.png"
            shutil.copy(filename[0], image_dest)
            # shutil.copy(mask_filename[0], mask_dest)

            mask_array = np.array(Image.open(mask_filename[0]))

            # Convert 255 to 1
            mask_array[mask_array == 255] = 1

            # Save the converted mask back to a file
            Image.fromarray(mask_array).save(mask_dest)
    else:
        image_dir = ["./data/ATLAS/splits_over_1/atlas_val_png.csv"]
        label_dir = ["./data/ATLAS/splits_over_1/atlas_val_mask_png.csv"]

        mask_dir = ["./data/ATLAS/splits_over_1/atlas_val_brain_mask_png.csv"]

        dataset = mask_preprocessing_loader(
            image_dir,
            label_dir=label_dir,
            mask_dir=mask_dir,
            target_size=(128, 128),
            test=False,
        )

        loader = DataLoader(
            dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True
        )

        for data in loader:
            filename = data[3]
            idx = data[2]

            index = int(filename[0].split("_")[-1].replace(".png", ""))
            image_dest = f"./nnunet_data/nnunet_raw/Dataset500_ATLAS/imagesTs/ATLAS_{index:03d}_0000.png"
            shutil.copy(filename[0], image_dest)
