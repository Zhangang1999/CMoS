import json
import time
from typing import Dict

from .file_manager import FileManager
from ..utils.time_utils import get_time_str

class LogManager(object):
    SESSION = [
        'train',
        'debug',
        'status',
    ]
    def __init__(self, file_manager:FileManager) -> None:
        super().__init__()

        self.file = file_manager

    def init_log(self, model_name:str, data:Dict={}) -> None:
        info = dict(
            type='header',
            data=data,
            time=get_time_str()
        )
        for sess in self.SESSION:
            self.file.create(model_name, {'log': f'{sess}.log'}, info)

    @property
    def file_manager(self) -> FileManager:
        return self.file
    @file_manager.setter
    def set_file_manager(self, new_file:FileManager) -> None:
        assert isinstance(new_file, FileManager), \
            f"file manager should FileManager, but got {type(new_file)}"
        self.file = new_file

    def log(self, model_name:str, session:str='', data:Dict={}) -> None:
        """log info to the log file belongs to model.

        Args:
            model_name (str): the target model name
            session (str, optional): the log type. Defaults to ''.
            data (Dict, optional): sth you want to log. Defaults to {}.
        """
        assert session in LogManager.SESSION, \
            f"session should in {LogManager.SESSION}, but got{session}"
        info = dict(
            data=data,
            time=get_time_str()
        )
    
        out = json.dumps(info) + '\n'
        with open(self.file.logs(model_name)[session], 'a') as f:
            f.write(out)    
