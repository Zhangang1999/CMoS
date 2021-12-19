
from os.path import join

class PathManager(object):

    def __init__(self, root:str) -> None:
        super().__init__()
        self._root = root
    
    @property
    def root(self):
        return self._root
    @root.setter
    def set_root(self, root):
        self._root = root
    
    def ckpt(self, model_name:str):
        return join(self._root, 'assets', 'runs', model_name, 'weights')
    
    def metric(self, model_name:str):
        return join(self._root, 'assets', 'runs', model_name, 'metrics')

    def visual(self, model_name:str):
        return join(self._root, 'assets', 'runs', model_name, 'visuals')

    def log(self, model_name:str):
        return join(self._root, 'assets', 'runs', model_name, 'logs')
  