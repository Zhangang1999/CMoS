
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.instantiate import instantiate_from_args

from ..backbones import BACKBONES
from ..heads import HEADS
from ..losses import LOSSES
from ..postprocess import POSTPROCESSES
from .mos_builder import MOS


@MOS.register()
class MMSA(object):

    def __init__(self, cfg, device, **kwargs) -> None:
        super().__init__()
        
        self.backbone = instantiate_from_args(cfg.backbone, BACKBONES).to(device)
        self.heads = instantiate_from_args(cfg.head, HEADS).to(device)
        self.loss = instantiate_from_args(cfg.loss, LOSSES).to(device)
        self.postprocess = instantiate_from_args(cfg.postprocess, POSTPROCESSES).to(device)
        
        self.device = device

    def forward(self, data):
        data = data.to(self.device)
        features = self.backbone(data)
        predicts = self.heads(features)
        return predicts

    def train_step(self, data, gts, optimizer=None):
        predicts = self.forward(data)
        losses = self.loss(predicts, gts)
        return dict(losses=losses, gts=gts, predicts=predicts)

    def valid_step(self, data, gts, optimizer=None):
        predicts = self.forward(data)
        results = self.postprocess(predicts)
        return dict(predicts=predicts, results=results, gts=gts)            

    def test_step(self, data, gts, optimizer=None):
        return self.valid_step(data, gts, optimizer)

    def infer_step(self, data):
        data = data.to(self.device)
        results = self.postprocess(self.forward(data))
        return results
