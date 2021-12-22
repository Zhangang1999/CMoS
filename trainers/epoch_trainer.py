
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
    
    def __init__(self, model, optimizer=None, file_manager=None, train_params: Dict = {}) -> None:
        """Trainer in epoch fashion.

        Args:
            model (callable[MODELS]): the model to train
            optimizer (callable[OPTIMIZER]): the optimizero used to train. Defaults to None.
            logger (callable[LOGGER]): the logger to log the info during train. Defaults to None.
            train_params (Dict): with some indicated keys. Defaults to {}.
        """
        super().__init__(model, optimizer=optimizer, file_manager=file_manager, train_params=train_params)

    def run_iter(self, datum, train_mode, **kwargs):
        """run the iteration step.

        Args:
            datum (Data): set of input data
            train_mode (str): the training mode. 'train' or 'valid'
        """
        if train_mode:
            outputs = self.model.train_step(*datum, self.optimizer, **kwargs)
        else:
            outputs = self.model.valid_step(*datum, self.optimizer, **kwargs)
        
        if not isinstance(outputs, dict):
            raise TypeError(f"outputs should be a dict, "
                            f"but got {type(outputs)}")

        self.outputs = outputs
    
    def train(self, data_loader, **kwargs):
        """the train procedure

        Args:
            data_loader (Lodaer): as name.
        """
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
        """the valid procedure.

        Args:
            data_loader (Loader): as name.
        """
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
        """the run procedure by the loaders and flow.

        Args:
            data_loaders (List[Lodaer]): list of data loaders
            workflow (List[str]): the procedure of workflow
        """
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

        try:
            while self._epoch < self._max_epochs:
                for i, flow in enumerate(workflow):
                    mode, epochs = flow
                    epoch_runner = getattr(self, mode)

                    for _ in range(epochs):
                        if mode == 'train' and self._epoch >= self._max_epochs:
                            break
                        epoch_runner(data_loaders[i], **kwargs) 
        except KeyboardInterrupt:
            self.save_checkpoint("interrupt")

        time.sleep(1)
        self.call_hook('after_run')             

    def save_checkpoint(self, status='latest', meta=None):
        """save the checkpoint for the model.

        Args:
            dst_dir (str): dstiniation of the savedir
            filename_tmpl (str): the template for the save file.
            meta (dict, optional): some meta infomation store here. Defaults to None.
        """
        filename_tmpl = self.model.name + '_ep{}_{}.pt'
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(f"meta should be a dict, "
                            f"but got {type(meta)}")
        meta.update(time=get_time_str())

        filename = filename_tmpl.format(self._epoch+1, status)
        filepath = os.path.join(self.file_manager.path.ckpt(self.model.name), filename)

        checkpoint = dict(
            state_dict=self.model.state_dict(),
            meta=meta
        )
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, status, strict=False, revise_keys=[]):
        """Load the checkpoint for the model.

        Args:
            filename ([type]): Accept local filepath
            strict (bool, optional): Whether to allow different params for the model and
                checkpoint. Defaults to False.
            revise_keys (list, optional): A list of customized keywords to modify the
                state_dict in checkpoint. Each item is a (pattern, replacement)
                pair of the regular expression operations. Defaults to [].
        """

        ckpts = self.file_manager.ckpts(self.models.name)
        checkpoint = torch.load(ckpts[status])
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"No state_dict found.")
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        for p, r in revise_keys:
            state_dict = OrderedDict(
                {re.sub(p, r, k): v
                    for k, v in state_dict.items()})
        
        self.model.load(state_dict, strict=strict)
