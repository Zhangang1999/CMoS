
import re
import os
import logging
import time
from typing import Dict
from collections import OrderedDict

from utils.time_utils import get_time_str
from .base_trainer import TRAINERS, BaseTrainer

@TRAINERS.register()
class EpochTrainer(BaseTrainer):

    def __init__(self, model, optimizer=None, logger=None, train_params: Dict = {}) -> None:
        super().__init__(model, optimizer=optimizer, logger=logger, train_params=train_params)

    def run_iter(self, datum, train_mode, **kwargs):
        if train_mode:
            outputs = self.model.train_step(datum, self.optimizer, **kwargs)
        else:
            outputs = self.model.valid_step(datum, self.optimizer, **kwargs)
        
        if not isinstance(outputs, dict):
            raise TypeError(f"outputs should be a dict, "
                            f"but got {type(outputs)}")

        self.outputs = outputs
    
    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        
        self.call_hook('before_train_epoch')
        time.sleep(2)

        for i, datum in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(datum, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    def valid(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'valid'
        self.data_loader = data_loader

        self.call_hook("before_valid_epoch")
        time.sleep(2)

        for i, datum in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook("before_valid_iter")
            self.run_iter(datum, train_mode=False, **kwargs)
            self.call_hook("after_valid_iter")
        
        self.call_hook("after_valid_epoch")
    
    def run(self, data_loaders, workflow, **kwargs):
        
        assert isinstance(data_loaders, list)
        assert isinstance(workflow[0], tuple)
        assert len(data_loaders) == len(workflow)
        assert self._max_epochs is not None

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break
        
        logging.info('Start running ...')
        logging.info(f'Hooks info:\n{self.get_hook_info()}')
        logging.info(f'workflow: {workflow}, max: {self._max_epochs}')
        self.call_hook('before_run')

        while self._epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                epoch_runner = getattr(self, mode)

                for _ in range(epochs):
                    if mode == 'train' and self._epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs) 

        time.sleep(1)
        self.call_hook('after_run')             

    def save_checkpoint(self, dst_dir, filename_tmpl, meta=None):
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(f"meta should be a dict, "
                            f"but got {type(meta)}")
        meta.update(time=get_time_str())

        filename = filename_tmpl.format(self._epoch+1)
        filepath = os.path.join(dst_dir, filename)

        checkpoint = dict(
            state_dict=self.model.state_dict(),
            meta=meta
        )
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filename, strict=False, revise_keys=[]):
        """[summary]

        Args:
            filename ([type]): Accept local filepath
            strict (bool, optional): Whether to allow different params for the model and
                checkpoint. Defaults to False.
            revise_keys (list, optional): A list of customized keywords to modify the
                state_dict in checkpoint. Each item is a (pattern, replacement)
                pair of the regular expression operations. Defaults to [].

        Raises:
            RuntimeError: [description]
        """
        checkpoint = torch.load(filename)
        if not isinstance(checkpoint, dict):
            raise RuntimeError(
                f"No state_dict found in {filename}")
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        for p, r in revise_keys:
            state_dict = OrderedDict(
                {re.sub(p, r, k): v
                    for k, v in state_dict.items()})
        
        self.model.load(state_dict, strict=strict)
