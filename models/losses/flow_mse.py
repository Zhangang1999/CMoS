import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowMSELoss(nn.Module):

    def __init__(self, cfg) -> None:
        super().__init__()


def warp(self, x, flow):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W]
    flow: [B, 2, H, W]
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = torch.autograd.Variable(grid) + flow

    # scale grid to [-1, 1] 
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W-1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H-1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)        
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())
    
    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    
    return output * mask