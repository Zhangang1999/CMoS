
from torchvision.transforms.functional import to_pil_image

from datasets.transforms import TRANSFORMS, BaseTransform


@TRANSFORMS.register()
class ToPILImage(BaseTransform):

    def __init__(self, p=1, mode=None) -> None:
        super().__init__(p)
        self.t = lambda x: to_pil_image(x, mode)

    def __call__(self, data_sample):
        data_sample.data = self.t(data_sample.data)
        
        if data_sample.gts is not None and 'mask' in data_sample.gts:
            data_sample.gts['mask'] = self.t(data_sample.gts['mask'])

        return data_sample