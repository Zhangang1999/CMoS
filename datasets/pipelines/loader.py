import os
from datasets.pipelines import PIPELINES

@PIPELINES.register()
class Loader(object):

    def __init__(self, data_fmt, **kwargs):
        self.data_fmt = data_fmt

    def __call__(self, data_sample):
        
        #TODO: Load data.
        image_dir = os.path.join(data_sample.meta.path, data_sample.meta.name)
        
        read = lambda x: x
        image = read(image_dir)
        gts = read(image_dir)

        #TODO: Change dtype.

        data_sample.meta.ori_shape = image.shape[:2]
        data_sample.meta.cur_shape = image.shape[:2]

        data_sample.data = image
        data_sample.gts = gts

        return data_sample