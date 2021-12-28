import math
import random
import os
import numpy as np
from collections import defaultdict

from datasets import DATASETS, BaseDataset
from datasets import SampleMeta, DatasetMeta, DataSample


@DATASETS.register()
class TrainValidDataset(BaseDataset):

    def __init__(self, cfg, transform=None) -> None:
        super().__init__(cfg, transform=transform)

    def _init_dataset(self, cfg) -> DatasetMeta:

        split_ratio = cfg.split_ratio if hasattr(cfg, 'split_ratio') else 0.8

        data_path = os.path.join(cfg.data_path, 'data')
        data_list = os.listdir(data_path)
        train, valid = self.random_split(data_list, split_ratio, cfg.random_seed)
        
        data_infos = defaultdict(list)
        for subset_name, subset_ids in zip(['train', 'valid'], [train, valid]):
            for data in data_list[subset_ids]:
                name, fmt = data.split('.')
                sample_meta = SampleMeta(
                    file_name=name,
                    file_fmt=fmt,
                    abs_path=os.path.join(data_path, data)
                )
                data_sample = DataSample(meta=sample_meta)
                data_infos[subset_name].append(data_sample)

            setattr(self, subset_name, SubSet(
                DatasetMeta(abs_path=cfg.data_path, data_infos=data_infos[subset_name])))

        return DatasetMeta(abs_path=cfg.data_path, data_infos=data_infos)
        
    @staticmethod
    def random_split(data_list, split_ratio, seed=42):
        random.seed(seed)

        index = list(range(len(data_list)))
        num_train = math.ceil(len(data_list) * split_ratio)
        train_index = random.choice(index, num_train)
        valid_index = set(index) - set(train_index)
        return list(train_index), list(valid_index)

class SubSet(BaseDataset):

    def __init__(self, meta, transform) -> None:
        super().__init__(meta, transform=transform)

    def _init_dataset(self, meta) -> DatasetMeta:
        return meta