
from torch.utils.data import DataLoader

from .loader_builder import LOADERS

@LOADERS.register()
class SimpleDataLoader(DataLoader):

    def __init__(self,
                 dataset,
                 batch_size,
                 shuffle,
                 num_workers=4,
                 pin_memory=False,
                 **kwargs
                 ) -> None:
       
        sampler = None,
        collate_fn = None,
        worker_init_fn = None,
        super().__init__(dataset=dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         sampler=sampler,
                         num_workers=num_workers,
                         collate_fn=collate_fn,
                         pin_memory=pin_memory,
                         worker_init_fn=worker_init_fn,
                         **kwargs
                         )
