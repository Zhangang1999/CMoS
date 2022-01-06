from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss_builder import LOSSES


@LOSSES.register()
class EPELoss(nn.Module):

    def __init__(self, cfg) -> None:
        """Supervised Flow loss from flownet.

        Args:
            cfg (configs): configs of the loss.
        """
        super().__init__()

        self.sparse = cfg.sparse
        self.mean = cfg.mean
        self.weights = cfg.weights

        self.loss_labels = cfg.loss_labels

    def forward(self, preds:Dict, gts:Dict) -> Dict:
        flows = preds['flow']

        losses = {}

        if 'MS' in self.loss_labels:
            losses['MS'] = self._multiscale_EPE(flows, gts)

        if 'R' in self.loss_labels:
            losses['R'] = self._real_EPE(flows[0], gts)    

        return losses    

    def _EPE(self, x, y):
        epe_map = torch.norm(y - x, 2, 1)
        batch_size = epe_map.size(0)

        if self.sparse:
            mask = (y[:, 0] == 0) & (y[:, 1] == 0)
            epe_map = epe_map[~mask]
        
        if self.mean:
            return epe_map.mean()
        else:
            return epe_map / batch_size

    def _multiscale_EPE(self, xs, y):
        def one_scale(x, y):
            b, _, h, w = x.size()

            if self.sparse:
                y_scaled = self.sparse_max_pool(y, (h, w))
            else:
                y_scaled = F.interpolate(y, (h, w), mode='area')
            return self._EPE(x, y_scaled)

        if type(xs) not in [tuple, list]:
            xs = [xs]
        if self.weights is None:
            self.weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
        assert(len(self.weights) == len(xs))

        loss = 0
        for x, weight in zip(xs, self.weights):
            loss += weight * one_scale(x, y)
        return loss

    def _real_EPE(self, x, y, sparse=False):
        _, _, h, w = y.size()
        upsampled_x = F.interpolate(x, (h, w), mode='bilinear', align_corners=False)
        return self._EPE(upsampled_x, y)

    @staticmethod
    def sparse_max_pool(x, size):
        '''Downsample the input by considering 0 values as invalid.

        Unfortunately, no generic interpolation mode can resize a sparse map correctly,
        the strategy here is to use max pooling for positive values and "min pooling"
        for negative values, the two results are then summed.
        This technique allows sparsity to be minized, contrary to nearest interpolation,
        which could potentially lose information for isolated data points.
        '''

        positive = (x > 0).float()
        negative = (x < 0).float()
        output = F.adaptive_max_pool2d(x * positive, size) \
               - F.adaptive_max_pool2d(-x * negative, size)
        return output
