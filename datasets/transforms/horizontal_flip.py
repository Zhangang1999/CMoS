
from .base_transform import TRANSFORMS, BaseTransform

from random import random
from torchvision.transform.functional import to_pil_image

import numpy as np

@TRANSFORMS.register()
class HorizonalFlip(BaseTransform):

    def __init__(self, p: float) -> None:
        super().__init__(p)
        self.t = lambda x: x[:, ::-1, :]

    def __call__(self, data_sample):
        if random() > self.p:
            return

        image = np.array(data_sample.data)
        new_image = self.t(image)
        data_sample.data = to_pil_image(new_image)

        if data_sample.gts is not None and 'mask' in data_sample.gts:
            new_mask = self.t(data_sample.gts['mask'])
            data_sample.gts['mask'] = new_mask
        
        return data_sample