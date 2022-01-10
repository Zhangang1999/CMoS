
from .base_transform import BaseTransform, TRANSFORMS

from torchvision.transforms.transforms import ColorJitter as tv_ColorJitter


@TRANSFORMS.register()
class Contrast(BaseTransform):

    def __init__(self, p:float, contrast:float) -> None:
        super().__init__(p)
        self.t = tv_ColorJitter(contrast=contrast)

    def __call__(self, data_sample):
        data_sample.data = self.t(data_sample.data)
        return data_sample