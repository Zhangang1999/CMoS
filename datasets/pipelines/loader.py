import os
from utils.image_io_utils import cv2_imread, imageio_imread
from .pipeline_builder import PIPELINES

class BaseLoader(object):
    HANDLER = {
        "cv2": imageio_imread,
        "imageio": cv2_imread,
    }
    def __init__(self, handler) -> None:
        super().__init__()
        assert handler in self.HANDLER        
        self.img_reader = self.HANDLER[handler]

    def __call__(self, data_sample):
        raise NotImplementedError

@PIPELINES.register()
class WithoutGTLoader(BaseLoader):

    def __init__(self, handler="cv2") -> None:
        super().__init__(handler)
    
    def __call__(self, data_sample):
        image_dir = os.path.join(data_sample.meta.path, data_sample.meta.name)

        image = self.img_reader(image_dir)

        data_sample.meta.ori_shape = image.shape[:2]
        data_sample.meta.cur_shape = image.shape[:2]

        data_sample.data = image

        return data_sample

@PIPELINES.register()
class WithImageGTLoader(BaseLoader):

    def __init__(self, handler='cv2') -> None:
        super().__init__(handler)

    def __call__(self, data_sample):
    
        image_dir = os.path.join(data_sample.meta.path, data_sample.meta.name)
        
        image = self.img_reader(image_dir)
        gts = self.img_reader(image_dir)

        data_sample.meta.ori_shape = image.shape[:2]
        data_sample.meta.cur_shape = image.shape[:2]

        data_sample.data = image
        data_sample.gts = gts

        return data_sample
