import torch
image_dir = './data/ATLAS/splits/atlas_train_png.csv'
mask_dir = './data/ATLAS/splits/atlas_train_mask_png.csv'

from torch.utils.data import DataLoader

from data.loaders.ixi_loader import mask_preprocessing_loader

dataset = mask_preprocessing_loader(image_dir, file_type='*.png', label_dir=mask_dir, target_size=(128, 128), test=False)
loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=4,
            shuffle=False,
            drop_last=False,
            pin_memory=True)

device = 'gpu' if torch.cuda.is_available() else 'cpu'
for data in loader:
    images = data[0].to(device)
    #masks = data[1].to(device)
    #print(images.shape, masks.shape)
    #index = data[2].to(device)
    #print(index)