import torch
from pytorch_msssim import ssim


l1 = torch.nn.L1Loss()

def combined_loss(output, target, alpha=0.5):
    l1_loss = l1(output, target)
    ssim_loss = 1 - ssim(
        torch.clamp(output, 0, 1),
        target,
        data_range=1.0,
        size_average=True
    )
    return l1_loss + alpha * ssim_loss
