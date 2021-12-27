
from abc import ABCMeta

from torch.utils.data import Dataset

from datasets.meta_desc import DatasetMeta


class BaseDataset(Dataset, metaclass=ABCMeta):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset_meta = self._init_dataset()

    def _init_dataset(self) -> DatasetMeta:
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset_meta.data_infos)
    