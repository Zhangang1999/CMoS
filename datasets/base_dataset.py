from typing import List
from abc import abcmeta, abstractmethod

from torch.utils.data import Dataset
from datasets.pipelines import PIPELINES
from utils.instantiate import instantiate_from_args
from datasets.meta_desc import DatasetMeta

class BaseDataset(Dataset, metaclass=abcmeta):

    def __init__(self, cfg, transform:List=[]) -> None:
        super().__init__()
        
        for worker, args in cfg.pipelines:
            setattr(self, worker, instantiate_from_args(args, PIPELINES))
        self.transform = self.composer([self.fetcher, transform, self.loader])

        self.dataset_meta = self._init_dataset(cfg)

    @abstractmethod
    def _init_dataset(self, cfg) -> DatasetMeta:
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset_meta.data_infos)

    def get_data_meta(self):
        data_meta = [data for data in self.dataset_meta.data_infos]
        return data_meta

    def get_item(self, i):
        item = self.dataset_meta.data_infos[i]
        item = self.transform(item)
        return item

    def __getitem__(self, i):
        return self.get_item(i)