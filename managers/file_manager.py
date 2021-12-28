import glob
import json
import logging
import os
from typing import Dict, List

from utils.time_utils import get_time_str

class FileManager(object):
    IN_MANAGEMENT = [
        'log',
        'ckpt',
        'metric',
        'visual'
    ]

    LOG_SESSION = [
        'train',
        'debug',
        'status',
    ]

    CKPT_STATUS = [
        'latest',
        'best',
        'interrupt'
    ]

    def __init__(self, root:str='./') -> None:
        """Manager the files & paths corresponding to the model
        for experiments.

        Args:
            root (str, optional): the root dir of the repo.. Defaults to './'.
        """
        super().__init__()
        self._root = os.path.abspath(root)
        self._model_name = None

    @property
    def root(self) -> str:
        return self._root
    @root.setter
    def set_root(self, root) -> None:
        self._root = root

    @property
    def model_name(self) -> str:
        return self._model_name
    @model_name.setter
    def model_name(self, name):
        self._model_name = name
    
    # PATHS, path corresponding to the model name.
    def ckpt(self, model_name:str=None) -> str:
        model_name = self._check_model_name(model_name)
        return os.path.join(self._root, 'assets', 'runs', model_name, 'weights')
    
    def metric(self, model_name:str=None) -> str:
        model_name = self._check_model_name(model_name)
        return os.path.join(self._root, 'assets', 'runs', model_name, 'metrics')

    def visual(self, model_name:str=None) -> str:
        model_name = self._check_model_name(model_name)
        return os.path.join(self._root, 'assets', 'runs', model_name, 'visuals')

    def log(self, model_name:str=None) -> str:
        model_name = self._check_model_name(model_name)
        return os.path.join(self._root, 'assets', 'runs', model_name, 'logs')

    # FILES, a dict corresponding to model name
    def logs(self, model_name:str=None) -> Dict:
        model_name = self._check_model_name(model_name)

        logs = {}
        for sess in self.LOG_SESSION:
            logs[sess] = glob.glob(self.log(model_name)+f'/*{sess}*')[0]
        return logs

    def ckpts(self, model_name:str=None) -> Dict:
        model_name = self._check_model_name(model_name)

        ckpts = {}
        for status in self.CKPT_STATUS:
            ckpts[status] = glob.glob(self.ckpt(model_name)+f'/*{status}*')[0]
        return ckpts

    def metrics(self, model_name:str=None) -> Dict:
        model_name = self._check_model_name(model_name)

        metrics = {}
        for file in os.listdir(self.metric(model_name)):
            metric = os.path.basename(file).split('_')[0]
            metrics[metric] = file
        return metrics

    def visuals(self, model_name:str=None) -> Dict:
        model_name = self._check_model_name(model_name)

        visuals = {}
        for file in os.listdir(self.metric(model_name)):
            visual_type = os.path.basename(file).split('_')[0]
            visuals[visual_type] = file
        return visuals

    # ACTIONS
    def log_init(self, model_name:str=None, data:Dict={}) -> None:
        """Initialize the log with given data,

        Args:
            model_name (str): the model name.
            data (dict, optional): header_data. Defaults to {}.
        """
        model_name = self._check_model_name(model_name)

        info = dict(
            type='header',
            data=data,
            time=get_time_str()
        )
        for sess in self.LOG_SESSION:
            self.create(model_name, {'log': f'{sess}.log'}, info)

    def log_log(self, model_name:str=None, session:str='', data:Dict={}) -> None:
        """log info to the log file belongs to model.

        Args:
            model_name (str): the target model name
            session (str, optional): the log type. Defaults to ''.
            data (Dict, optional): sth you want to log. Defaults to {}.
        """
        model_name = self._check_model_name(model_name)

        assert session in self.LOG_SESSION, \
            f"session should in {self.LOG_SESSION}, but got{session}"
        info = dict(
            data=data,
            time=get_time_str()
        )
    
        out = json.dumps(info) + '\n'
        with open(self.logs(model_name)[session], 'a') as f:
            f.write(out)    

    # MAKE DIRS & FILES
    def create(self, model_name:str=None, data:Dict={}, extra_info:Dict={}) -> None:
        """create files for the model according to data.

        Args:
            model_name (str): the target model name
            data (Dict): dict of key-type and value-filename.
            extra_info (Dict): sth you want to write in file.

        Raises:
            ValueError: if file type not in managerment.
        """
        for k, v in data.items():
            if k not in FileManager.IN_MANAGEMENT:
                raise ValueError(f'to create file should in {FileManager.IN_MANAGEMENT}')
            
            self._check_if_exist_and_make(getattr(self, k)(model_name))
            fp = os.path.join(getattr(self, k)(model_name), v)

            out = json.dumps(extra_info) + '\n'
            with open(fp, 'w') as f:
                f.write(out)
            logging.info('create {k} file: {fp}')
    
    def makedirs(self, model_name:str=None, exclude_dir:List[str]=[]) -> None:
        """Making dirs corresponding the model name.

        Args:
            model_name (str): the model name.
            exclude_dir (list, optional): the dir not to make.. Defaults to [].
        """
        model_name = self._check_model_name(model_name)

        for k in self.IN_MANAGEMENT:
            if k in exclude_dir: continue
            self._check_if_exist_and_make(getattr(self, k)(model_name))
            logging.info(f"make dirs for {model_name}-{k}.")

    @staticmethod
    def _check_if_exist_and_make(path:str) -> None:
        """Check the path if exist, if not, make it.

        Args:
            path (str): the dir path to make.
        """
        if not os.path.exists(path):
            os.makedirs(path)
            logging.info(f"create path: {path}.")

    def _check_model_name(self, model_name):
        if model_name is None:
            assert self._model_name is not None
            model_name = self._model_name
        return model_name