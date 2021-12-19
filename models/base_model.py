
import torch
import torch.nn as nn

class BaseModel(nn.Module):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
    
    def forward(self, x):
        return x

    def init_weights(self, init_type='he2'):
        """ Initialize weights for training. """

        init_func = {
            'he1': lambda w: nn.init.kaiming_normal_(w, a=1e-2, nonlinearity='leaky_relu'),
            'he2': lambda w: nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu'),
            'xia': lambda w: nn.init.xavier_uniform_(w, gain=1),
        }

        # Initialize the weights.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_func[init_type](m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super().train(mode)

        if self.cfg.freeze_bn:
            self.freeze_bn()

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable