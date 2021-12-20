from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from spatial_correlation_sampler import spatial_correlation_sample
from backbones import BACKBONES

@BACKBONES.register()
class FlowNetS(nn.Module):
    expansion = 1
    def __init__(self, cfg) -> None:
        super().__init__()

        self.encoder_layers = self._build(cfg.encoder, conv)
        self.decoder_layers = self._build(cfg.decoder, deconv)
        self.predict_layers = self._build(cfg.predict, predict)
        self.upsample_layers = self._build(cfg.upsample, nn.ConvTranspose2d)

        self._init()

    def _build(self, layer_cfgs, layer_func):
        layers = OrderedDict()
        for scope, layer_args in layer_cfgs.items():
            layer = sum([layer_func(*args) for args in layer_args], [])
            layers.update({f'{scope}': layer})
        return nn.ModuleDict(layers)

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        
        encs = OrderedDict()
        for scope, layer in self.encoder_layers.items():
            x = layer(x)
            encs[scope] = x

        enc_scopes = list(encs.keys())[::-1]
        last_scope = enc_scopes.pop(-1)

        dec, flow_up = None, None
        flows = []
        for scope in enc_scopes:
            flow = self.predict_layers[scope](
                concat([encs[scope], dec, flow_up]))
            flow_up = self.upsample_layers[scope](flow)
            dec = self.decoder_layers[scope](encs[scope])
            flows.append(flows)

        flow = self.predict_layers[last_scope](
            concat([encs[scope], dec, flow_up]))
        flows.append(flow)

        pred_outs = {}
        if self.training:
            pred_outs['flow'] = flows[::-1]
        else:
            pred_outs['flow'] = flow
        
        return pred_outs

@BACKBONES.register()
class FlowNetC(FlowNetS):
    expansion = 1
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.pre_encoder_layers = self._build(cfg.pre_encoder, conv)
        self.redir_layer = self._build(cfg.redir_layer, conv)

    def forward(self, x):
        x1, x2 = x[:, :3], x[:, 3:]

        for scope, layer in self.pre_encoder_layers.items():
            x1 = layer(x1)
            x2 = layer(x2)
        x = torch.cat([self.redir_layer(x1), correlate(x1, x2)], 1)           

        encs = OrderedDict()
        for scope, layer in self.encoder_layers.items():
            x = layer(x)
            encs[scope] = x

        enc_scopes = list(encs.keys())[::-1]
        last_scope = enc_scopes.pop(-1)
       
        dec, flow_up = None, None
        flows = []
        for scope in enc_scopes:
            flow = self.predict_layers[scope](
                concat([encs[scope], dec, flow_up]))
            flow_up = self.upsample_layers[scope](flow)
            dec = self.decoder_layers[scope](encs[scope])
            flows.append(flows)

        flow = self.predict_layers[last_scope](
            concat([encs[scope], dec, flow_up]))
        flows.append(flow)

        pred_outs = {}
        if self.training:
            pred_outs['flow'] = flows[::-1]
        else:
            pred_outs['flow'] = flow
        
        return pred_outs

# -----------------------------------------------------------
# Operation functions below

def concat(feats, dim):
    assert isinstance(feats, list)

    feats = [feat for feat in feats if feat]
    return torch.cat(*feats, dim)

def conv(use_bn, 
         in_channels, 
         out_channels, 
         kernel_size=3, 
         stride=1):

    if use_bn:
        return nn.Sequential(
            nn.Conv2d(in_channels, 
                      out_channels, 
                      kernel_size=kernel_size, stride=stride, 
                      padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels, 
                      kernel_size=kernel_size, stride=stride, 
                      padding=(kernel_size-1)//2,
                      bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        ) 

def predict(in_channels):
    return nn.Conv2d(in_channels, 2, 
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     bias=False)

def deconv(in_channels, out_channels):
    return nn.Sequential(
           nn.ConvTranspose2d(in_channels,
                              out_channels, 
                              kernel_size=4, 
                              stride=2, 
                              padding=1, 
                              bias=False),
           nn.LeakyReLU(0.1, inplace=True)
    )

def correlate(x1, x2):
    out_corr = spatial_correlation_sample(x1, x2,
                                          kernel_size=1,
                                          patch_size=21,
                                          stride=1,
                                          padding=0,
                                          dilation_patch=2)
    # collate dimensions 1 and 2 in order to be treated as a
    # regular 4D tensor
    b, ph, pw, h, w = out_corr.size()
    out_corr = out_corr.view(b, ph * pw, h, w) / x1.size(1)
    return F.leaky_relu_(out_corr, 0.1)
