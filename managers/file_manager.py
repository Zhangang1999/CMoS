
import os
import glob
import json
import logging
from typing import Dict

from managers.path_manager import PathManager

class FileManager(object):
    IN_MANAGEMENT = [
        'log',
        'ckpt',
        'metric',
        'visual'
    ]

    def __init__(self, path:PathManager) -> None:
        super().__init__()

        self.path = path

    @property
    def path_manager(self) -> PathManager:
        return self.path
    @path_manager.setter
    def set_path_manager(self, new_path:PathManager) -> None:
        assert isinstance(new_path, PathManager), \
            f"path manager should PathManager, but got {type(new_path)}"
        self.path = new_path
    
    def logs(self, model_name:str) -> Dict:
        """Return a dict of all logs belongs to {MODEL}.

        Args:
            model_name (str): the target model name

        Returns:
            Dict: {kep:fp,}
        """
        logs = {}
        for sess in ('train', 'debug', 'status'):
            logs[sess] = glob.glob(self.path.log(model_name)+f'/*{sess}*')[0]
        return logs

    def ckpts(self, model_name:str):
        # TODO: return a dict according to phase. ['best', 'latest', 'interrupt'.]
        return os.listdir(self.path.ckpt(model_name))

    def metrics(self, model_name:str):
        return os.listdir(self.path.metric(model_name))

    def create(self, model_name:str, data:Dict, extra_info:Dict) -> None:
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
            
            self._check_if_exist(getattr(self.path, k)(model_name))
            fp = os.path.join(getattr(self.path, k)(model_name), v)
        
            out = json.dumps(extra_info) + '\n'
            with open(fp, 'w') as f:
                f.write(out)
            logging.info('create {k} file: {fp}')
                
    @staticmethod
    def _check_if_exist(path) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
            logging.info(f"create path: {path}.")
