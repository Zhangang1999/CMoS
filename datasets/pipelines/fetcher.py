
from logging import NOTSET
from datasets.pipelines import PIPELINES


class BaseFetcher(object):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data_sample):
        raise NotImplementedError
        

@PIPELINES.register()
class TupleFetcher(BaseFetcher):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data_sample):

        image = data_sample.data

        if data_sample.gts is not None:
            gts = {k:v for k,v in data_sample.gts.items()}
            return image, gts

        return image

@PIPELINES.register()
class DictFetcher(BaseFetcher):

    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, data_sample):

        data_dict = {}
        data_dict['image'] = data_sample.data

        if data_sample.gts is not None:
            data_dict.update({k:v for k,v in data_sample.gts.items()})
        return data_dict