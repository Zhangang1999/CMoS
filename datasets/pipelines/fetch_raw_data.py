
from logging import NOTSET

from datasets.pipelines import PIPELINES

@PIPELINES.register()
class FetchRawData(object):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data_sample):

        image = data_sample.data

        if data_sample.gts is not None:
            gts = {k:v for k,v in data_sample.gts.items()}
            return image, gts

        return image
