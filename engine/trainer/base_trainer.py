
from abc import ABCMeta, abstractmethod
from typing import Dict

from managers.ops_manager import OpsManager
from managers.log_manager import LogManager
from torch.optim import Optimizer

from hooks.base_hook import BaseHook

from ...utils.distribute_utils import get_dist_info
from ...utils.time_utils import get_time_str

TRAINERS = OpsManager('trainer')

class BaseTrainer(metaclass=ABCMeta):

    def __init__(self,
                 model,
                 optimizer=None,
                 logger=None,
                 train_params:Dict={},
                 ) -> None:
        
        self._check_optimizer_if_valid(optimizer)
        self._check_logger_if_valid(logger)

        self.model = model
        self.optimizer = optimizer
        self.logger = logger

        self._rank, self._world_size = get_dist_info()
        self.timestamp = get_time_str()

        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0

        self._check_train_params_if_valid(train_params)

    def _check_train_params_if_valid(self, train_params):
        if not isinstance(train_params, dict):
            raise TypeError(f"train_params should be a dict, "
                            f"but got {type(train_params)}")
        if 'max_iters' not in train_params and 'max_epochs' not in train_params:
            raise ValueError("Only one of `max_iters` or `max_epochs` can be set")
        
        self._max_iters = train_params['max_iters']
        self._max_epochs = train_params['max_epochs']

    def _check_optimizer_if_valid(self, optimizer):
        if isinstance(optimizer, dict):
            for name, optim in optimizer.items():
                if not isinstance(optim, Optimizer):
                    raise TypeError(f"optimizer should be a dict of torch.optim.Optimizer, "
                                    f"but got {name} is a {type(optim)}")
        elif not isinstance(optimizer, Optimizer) and optimizer is not None:
            raise TypeError(
                f"optimizer should be a torch.optim.Optimizer object, "
                f"but got {type(optimizer)}")
    
    def _check_logger_if_valid(self, logger):
        if not isinstance(logger, LogManager):
            raise TypeError(f"logger should be a Logger object, "
                            f"but got {type(logger)}")

    @property
    def model_name(self):
        if hasattr(self.model, 'module'):
            return self.model.module.__class__.__name__
        else:
            return self.mode.__class__.__name__
    
    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def hooks(self):
        return self._hooks

    @property
    def iter(self):
        return self._iter
    
    @property
    def inner_iter(self):
        return self._inner_iter
    
    @property
    def max_epochs(self):
        return self._max_epochs()
    
    @property
    def max_iters(self):
        return self._max_iters()

    @abstractmethod
    def train(self, data_loader, **kwargs):
        pass

    @abstractmethod
    def valid(self, data_loader, **kwargs):
        pass

    @abstractmethod
    def run(self, data_loaders, workflow, **kwargs):
        pass

    def save_checkpoint(self, dst_dir, filename_tmpl):
        pass

    @abstractmethod
    def load_checkpoint(self, filename, strict=False):
        pass

    def current_lr(self):
        if isinstance(self.optimizer, Optimizer):
            lr = [group['lr'] for group in self.optimizer.param_groups]
        elif isinstance(self.optimizer, dict):
            lr = dict()
            for name, optim in self.optimizer.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                "lr is not applicable because optimizer does not exist.")
        return lr
    
    def register_hook(self, hook, priority):
        assert isinstance(hook, BaseHook), 'hook sholud be an instance of HOOK.'

        hook.priority = priority

        inserted = False
        for i in range(len(self._hooks)-1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i+1, hook)
                inserted = True
        if not inserted:
            self._hooks.insert(0, hook)

    def call_hook(self, stage):
        for hook in self._hooks:
            getattr(hook, stage)(self)
    
    def get_hook_info(self):
        stage_hook_map = {stage: [] for stage in BaseHook.STAGES}
        for hook in self.hooks:   
            priority = hook.priority
            class_name = hook.__class__.__name__
            hook_info = f"({priority:<12}) {class_name:<35}"
            for trigger_stage in hook.get_triggered_stages():
                stage_hook_map[trigger_stage].append(hook_info)
        
        stage_hook_infos = []
        for stage in BaseHook.STAGES:
            hook_infos = stage_hook_map[stage]
            if len(hook_infos) > 0:
                info = f"{stage}: \n"
                info += '\n'.join(hook_infos)
                info += '\n ---------------- '
                stage_hook_infos.append(info)
        return '\n'.join(stage_hook_infos)