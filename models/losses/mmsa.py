from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss_builder import LOSSES

@LOSSES.register()
class MMSALoss(nn.Module):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.loss_labels = cfg.loss_labels

    def forward(self, pred:Dict, gts:Dict) -> Dict:
        pred_flows = pred['flow']
        gt_flows = gts['flow']

        pred_cls = pred['cls']
        gt_cls = gts['cls']

        losses = {}

        if 'W' in self.loss_labels:
            losses['W'] = self._flow_warp_loss(pred_flows, gt_flows)
        
        if 'S' in self.loss_labels:
            losses['S'] = self._flow_smooth_loss(pred_flows, gt_flows)
            
        if 'C' in self.loss_labels:
            losses['C'] in self._motion_classify_loss(pred_cls, gt_cls)

    def _flow_warp_loss(self, xs, y):
        eps = 1e-8
        prev, post = torch.split(y, 2, 1)

        loss_warp = 0.
        for x in xs:
            prev = F.adaptive_avg_pool2d(prev, x.size())
            post = F.adaptive_avg_pool2d(post, x.size())

            prev_warp_to_post = warp(prev, x)
            post_warp_to_prev = warp(post, x.neg())

            prev_loss = torch.norm(prev-post_warp_to_prev+eps, p=2, dim=1)
            post_loss = torch.norm(post-prev_warp_to_post+eps, p=2, dim=1)

            loss_warp += 0.5 * (post_loss+prev_loss).mean()
        return loss_warp

    def _flow_smooth_loss(self, xs, y, gradient):
        loss_smooth = 0.
        for x in xs:
            u, v = torch.split(x, 2, 1)
            loss_smooth += (gradient(u)+gradient(v)).sqrt().mean()
        return loss_smooth

    def _motion_classify_loss(self, x, y):
        y_onehot = onehot(y.unsqueeze(-1).long())
        loss_ce = F.binary_cross_entropy_with_logits(
                    x.sigmoid(), y_onehot, reducion='none')
        return loss_ce.mean()

def onehot(x, labels, dim=1):
    assert x.dim() == labels.dim()
    return x.scatter_(dim, labels, 1)

@torch.no_grad()
def gradient_by_sobel(x):
    sobel = torch.FloatTensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, -1]]) / 6.
    sobel = sobel.to(x.device)

    sobel_x = sobel.unsqueeze(-1).unsqueeze(-1)
    sobel_y = sobel_x.permute(1, 0, 2, 3)

    g_x = F.conv2d(x, sobel_x)
    g_y = F.conv2d(x, sobel_y)
    return torch.add(g_x.pow(2), g_y.pow(2))

def gradient_by_pad(x):
    h, w = x.size()[2:]

    r = F.pad(x, (0,1,0,0))[:,:,:,1:]
    l = F.pad(x, (1,0,0,0))[:,:,:,:w]
    t = F.pad(x, (0,0,1,0))[:,:,:h,:]
    b = F.pad(x, (0,0,0,1))[:,:,1:,:]

    g_x = (r - l) * 0.5
    g_y = (t - b) * 0.5
    return torch.add(g_x.pow(2), g_y.pow(2))

def warp(x, flow):
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
    output = F.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).to(vgrid.device)
    mask = F.grid_sample(mask, vgrid)

    # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())
    
    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    
    return output * mask
