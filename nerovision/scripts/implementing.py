import numpy as np
import torch

from nerovision.dl.auto_encoding.models.CAE import CAE


def create_ae_parameters():
    parameters = {}
    return parameters


def main():
    ae_pm = create_ae_parameters()
    features = [32, 64, 128, 256]
    im_h = 224
    im_w = 224
    latent_dim = 1024
    ae = CAE(features=features, im_h=im_h, im_w=im_w, latent_dim=latent_dim)

    empty_im = np.zeros((1, 3, im_h, im_w), dtype=np.float32)
    empty_im = torch.tensor(empty_im)
    ae(empty_im)


if __name__ == '__main__':
    main()
