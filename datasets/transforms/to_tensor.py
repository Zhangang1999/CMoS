

from torchvision.transforms.functional import to_tensor
from datasets.transforms import TRANSFORMS, BaseTransform

@TRANSFORMS.register()
class ToTensor(BaseTransform):

    def __init__(self, p=1) -> None:
        super().__init__(p)
        self.t = lambda x: to_tensor(x)

    def __call__(self, data_sample):

        data_sample.data = self.t(data_sample.data)

        if data_sample.gts is not None:
            for key, gt in data_sample.gts.items():
                data_sample.gts[key] = self.t(gt)

        return data_sample