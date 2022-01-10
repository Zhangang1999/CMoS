from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .head_builder import HEADS

@HEADS.register()
class MMSAHead(nn.Module):

    def __init__(self, cfg) -> None:
        super().__init__()

        self.encoder_layers, out_channels = self._build(cfg.encoder, cfg.in_channels, conv)
        self.predict_layers, _ = self._build(cfg.predict, out_channels, pred)

        self._init()

    def _build(self, layer_cfgs:Dict, in_channels, layer_func):
        
        def build(args):
            nonlocal in_channels
            layer = layer_func(in_channels, *args)
            in_channels = args[0]
            return layer

        layers = OrderedDict()
        for scope, layer_args in layer_cfgs.items():
            layer = sum([build(args) for args in layer_args], [])
            layers.update({f'{scope}': layer})
        return nn.ModuleDict(layers), in_channels

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def conv(in_channels, 
         out_channels, 
         kernel_size=3, 
         stride=1,
         leaky=0.2,
         pool_cfg=None,
         msa_cfg=None,
         adc_cfg=None,
         tse_cfg=None,
         nlb_cfg=None,
         ):
    
    layers = []
    if adc_cfg is not None:
        layers.append(AdaptiveDepthConv(**adc_cfg))
        layers.append(ActvLayer(leaky, in_place=True))

    if msa_cfg is not None:
        layers.append(MotionSpatialAttention(**msa_cfg))

    layers.append(nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=(kernel_size-1)//2,
                            bias=True,
                            ))
    layers.append(ActvLayer(leaky, in_place=True))
    
    if tse_cfg is not None:
        layers.append(TemporalSqueezeExcitation(**tse_cfg))

    if nlb_cfg is not None:
        layers.append(NonLocalBlock(**tse_cfg))

    if pool_cfg is not None:
        layers.append(nn.AvgPool2d(**pool_cfg))
    
    return nn.Sequential(*layers)
    
def pred(in_channels,
         out_channels,
         dropout_cfg=None,
         ):

    layers = []
    if dropout_cfg is not None:
        layers.append(nn.Dropout(**dropout_cfg))
    layers.append(nn.Linear(in_channels, out_channels))
    return nn.Sequential(*layers)

class ActvLayer(nn.Module):

    def __init__(self, leaky) -> None:
        super().__init__()
        self.leaky = leaky

    def forward(self, x):
        return torch.max(x, x*0.2)

class MotionSpatialAttention(nn.Module):

    def __init__(self, 
                 in_channels,
                 out_channels,
                 flow_in_channels,
                 flow_out_channels,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.spatial_temporal_encoder = nn.Conv2d(in_channels+2, out_channels, kernel_size=1, stride=1, padding=0)
        self.motion_encoder = nn.Conv2d(flow_in_channels, flow_out_channels, kernel_size=1, stride=1, padding=0)
        self.attention_encoder = nn.Conv2d(flow_out_channels+out_channels, 1, kernel_size=1, stride=1, padding=0) 

    def forward(self, x, flow):
        
        b, _, h, w = x.size()
        flow = flow.unsqueeze(-1).reshape(-1, 2, b, h, w)
        flow = flow.flatten(0, 1).permute(1, 0, 2, 3)

        fea_spatial_temporal = self._spatial_temporal_encode(x, flow)
        fea_motion = self._motion_encode(flow)
        fea_attention = self.attention_encoder(fea_spatial_temporal, fea_motion)

        return torch.mul(x, fea_attention).add(x)

    def _motion_encode(self, flow):
        return self.motion_encoder(flow)

    def _spatial_temporal_encode(self, x, flow):

        avg_flow = flow.mean(dim=1, keepdims=True)
        max_flow = flow.abs().max(dim=1, keepdims=True)

        fea_spatial_temporal = torch.cat([x, avg_flow, max_flow], dim=1)
        return self.spatial_temporal_encoder(fea_spatial_temporal)

    def _attention_encode(self, fea_spatial_temporal, fea_motion):
        fea_fusion = torch.cat([fea_spatial_temporal, fea_motion], dim=1)
        return self.attention_encoder(fea_fusion).sigmoid()

class NonLocalBlock(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super().__init__()
        self.to_q = nn.Conv2d(in_channels, out_channels//2, kernel_size=1, stride=1, padding=0, **kwargs)
        self.to_k = nn.Conv2d(in_channels, out_channels//2, kernel_size=1, stride=1, padding=0, **kwargs)
        self.to_v = nn.Conv2d(in_channels, out_channels//2, kernel_size=1, stride=1, padding=0, **kwargs)

        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        b, _, h, w = x.size()

        q = self.to_q(x).permute(0, 2, 3, 1).flatten(1)
        k = self.to_k(x).permute(0, 2, 3, 1).flatten(1)
        v = self.to_v(x).permute(0, 2, 3, 1).flatten(1)

        att = torch.einsum('bc, bc -> cc', q, k).softmax(dim=1)
        out = torch.einsum('cc, bc -> bc', att, v).unsqueeze(0)
        out = out.reshape(1, b, h*w, -1).permute(0, 3, 1, 2)
        out = self.out_conv(out).add(x)

        return out.permute(2, 1, 0, 3).reshape(b, -1, h, w)

class TemporalSqueezeExcitation(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 squeeze_ratio,
                 min_channels=32,
                 **kwargs,
                 ) -> None:
        super().__init__()
        
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))

        hidden_channels = torch.max(out_channels / squeeze_ratio, min_channels)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=False),
            nn.ReLU(True),
            nn.Linear(hidden_channels, out_channels, bias=False),
        )

    def forward(self, x):
        z = self.squeeze(x)
        att = self.excitation(z).sigmoid()
        return torch.einsum('bchw, bc -> bchw', x, att)
    
class AdaptiveDepthConv(nn.Module):

    def __init__(self,
                 out_channels,
                 kernel_size,
                 N_min,
                 N_max,
                 **kwargs) -> None:
        super().__init__()
        self.N_max = N_max
        self.conv = nn.Conv2d(N_min, 
                              out_channels, 
                              kernel_size=kernel_size, 
                              stride=1, 
                              padding=0, 
                              bias=True)

    def forward(self, x):
        kernel = self.conv.weight
        self.conv.weight = self._bilinear_kernel_resample(kernel, x.size(1))
        return self.conv(x)

    @torch.no_grad()
    def _bilinear_kernel_resample(self, kernel, N):
        ch, N_min, xlen, ylen = kernel.size()
        kernel = kernel.permute(1, 0, 2, 3)

        zeros = torch.zeros([self.N_max-N_min, ch, xlen, ylen]).float().to(kernel.device)
        new_kernel = torch.cat([kernel, zeros], dim=0)
        new_kernel = new_kernel.reshape(self.N_max, -1)

        xx = torch.linspace(0, N_min-1, N)
        ones = torch.ones([self.N_max-N]).float().to(kernel.device)
        xx_ = torch.cat([xx, ones], dim=0)
        ind0 = torch.floor(xx_).long()
        ind1 = torch.ceil(xx_).long()

        w0 = (xx - torch.floor(xx)).reshape(N, 1)
        zeros = torch.zeros([self.N_max-N, 1]).float().to(kernel.device)
        w0 = torch.cat([w0, zeros], dim=0)
        w1 = 1. - w0
        w0 = torch.tile(w0, (1, new_kernel.size(-1)))
        w1 = torch.tile(w1, (1, new_kernel.size(-1)))

        new_kernel = w0 * torch.gather(new_kernel, ind1) + w1 * torch.gather(new_kernel, ind0)
        new_kernel = new_kernel * N_min / N

        return new_kernel.reshape(self.N_max, ch, xlen, ylen)
