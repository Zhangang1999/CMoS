import numpy as np
from abc import abcmeta

from torch.utils.data import Dataset
from datasets.pipelines import PIPELINES
from utils.instantiate import instantiate_from_args
from datasets.meta_desc import DatasetMeta

class BaseDataset(Dataset, metaclass=abcmeta):

    def __init__(self, cfg, pipeline=None) -> None:
        super().__init__()
        if pipeline is not None:
            self.transform = instantiate_from_args(cfg.composer, PIPELINES)(pipeline)
        self.dataset_meta = self._init_dataset()

    def _init_dataset(self) -> DatasetMeta:
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset_meta.data_infos)

    def get_gts(self):
        gts = np.array([data.gts for data in self.dataset_meta.data_infos])
        return gts

    def get_item(self, i):
        item = self.dataset_meta.data_infos[i]
        if hasattr(self, 'transform'):
            item = self.transform(item)
        return item

    def __getitem__(self, i):
        return self.get_item(i)