import os
import glob

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torch.utils.data import DataLoader

from nerovision.nv_enums import ImageFormat
from nerovision.dl.auto_encoding.data.FilepathAEDataset import FilepathAEDataset
from nerovision.dl.auto_encoding.training import AETrainer
from nerovision.dl.auto_encoding.models.CAE import CAE
from nerovision.dl.callbacks.Checkpointer import Checkpointer
from nerovision.dl.callbacks.CSVLogger import CSVLogger


def get_transformations(im_h: int, im_w: int, image_format: ImageFormat):
    if image_format == ImageFormat.rgb:
        normalize = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    elif image_format == ImageFormat.gray_16 or image_format == ImageFormat.gray_8:
        normalize = A.Normalize(mean=0.5, std=0.5)
    else:
        raise ValueError('Unsupported image format: {}'.format(image_format))

    t_train = A.Compose(
        [
            # A.SmallestMaxSize(max_size=160),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.2),
            A.RandomCrop(height=im_h * 2, width=im_w * 2, p=0.2),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.HorizontalFlip(p=0.2),
            A.Affine(p=0.2, scale=0.8, shear=5, translate_percent=0.1, rotate=20),
            A.Blur(blur_limit=3, p=0.1),
            A.OpticalDistortion(p=0.1),
            A.GridDistortion(p=0.1),
            A.HueSaturationValue(p=0.1),
            A.Resize(height=im_h, width=im_w, p=1),
            normalize,
            ToTensorV2(),
        ]
    )
    t_val = A.Compose(
        [
            A.Resize(height=im_h, width=im_w, p=1),
            normalize,
            ToTensorV2(),
        ]
    )
    return t_train, t_val


def create_loaders(
        dataset_path: str,
        im_h: int,
        im_w: int,
        im_format: ImageFormat,
        val_size: float = 0.2,
):
    train_transform, val_transform = get_transformations(im_h, im_w, im_format)
    im_paths = glob.glob(os.path.join(dataset_path, '**/**.png'), recursive=True)
    n = int(len(im_paths) * (1 - val_size))
    train_paths = im_paths[:n]
    val_paths = im_paths[n:]

    train_dataset = FilepathAEDataset(filepaths=train_paths, transformations=train_transform, im_format=im_format)
    val_dataset = FilepathAEDataset(filepaths=val_paths, transformations=val_transform, im_format=im_format)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, persistent_workers=True)
    return train_loader, val_loader


def main():
    features = [32, 64, 128, 256]
    im_h = 224
    im_w = 224
    latent_dim = 1024
    # create data loaders
    dataset_path = r'C:\dev\autoencoding\data\transistor\train'
    model_path = r'C:\dev\autoencoding\models\model.onnx'
    csv_log_path = r'C:\dev\autoencoding\models\logs.csv'
    train_loader, val_loader = create_loaders(
        dataset_path,
        im_h=im_h,
        im_w=im_w,
        im_format=ImageFormat.rgb,
    )

    ae = CAE(features=features, im_h=im_h, im_w=im_w, latent_dim=latent_dim)
    trainer = AETrainer(ae)
    callbacks = [
        Checkpointer(model_path, im_h, im_w, 3),
        CSVLogger(csv_log_path),
    ]
    trainer.train_model(
        epochs=500,
        train_loader=train_loader,
        val_loader=val_loader,
        callbacks=callbacks,
    )

    empty_im = np.zeros((1, 3, im_h, im_w), dtype=np.float32)
    empty_im = torch.tensor(empty_im)
    print('finished')


if __name__ == '__main__':
    main()
