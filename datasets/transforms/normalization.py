
from torchvision.transforms import Normalize as tv_Normalize

from .base_transform import TRANSFORMS, BaseTransform

@TRANSFORMS.register(BaseTransform)
class Normalization():

    def __init__(self, p=1, mean=0., std=1., **kwargs) -> None:
        super().__init__()
        self.t = tv_Normalize(mean=mean, std=std)

    def __call__(self, data_sample):
        data_sample.data = self.t(data_sample.data)
        return data_sample
        