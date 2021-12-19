
from typing import List, Tuple

ShapeType = Tuple[int, int]

class SampleMeta(object):

    def __init__(self, 
                 file_name:str,
                 file_fmt:str,
                 ori_shape:ShapeType,
                 cur_shape:ShapeType,
                 abs_path:str,
                 ) -> None:
        super().__init__()

        self._file_name = file_name
        self._file_fmt = file_fmt
        self._ori_shape = ori_shape
        self._cur_shape = cur_shape
        self._abs_path = abs_path

    @property
    def name(self):
        return self._file_name
    
    @property
    def format(self):
        return self._file_fmt
    
    @property
    def ori_shape(self):
        return self._ori_shape
    
    @property
    def cur_shape(self):
        return self._cur_shape

    @property
    def path(self):
        return self._abs_path

    def __repr__(self):
        repr_str = f"File: {self.name}{self.format}\n"
        repr_str += f"  Path: {self.path}\n"
        repr_str += f"  Original Shape: {self.ori_shape}\n"
        repr_str += f"  Current Shape: {self.cur_shape}\n"
        return repr_str

    def __hash__(self):
        return self.name.__hash__()

class DataSample(object):

    def __init__(self, 
                 data,
                 gt,
                 meta:SampleMeta,
                ) -> None:
        super().__init__()

        self._data = data
        self._gt = gt
        self._ori_gt = gt
        self._meta = meta
    
    @property
    def data(self):
        return self._data

    @property
    def gt(self):
        return self._gt
    
    @property
    def ori_gt(self):
        return self._ori_gt

    @property
    def meta(self):
        return self._meta
    
    def __repr__(self) -> str:
        return self.meta.__repr__()

class DatasetMeta(object):

    def __init__(self,
                 abs_path,
                 data_infos:List[DataSample],
                 ) -> None:
        super().__init__()

        self._abs_path = abs_path
        self._data_infos = data_infos
    
    def __len__(self):
        return len(self._data_infos)
    
    @property
    def path(self):
        return self._abs_path

    @property
    def data_infos(self):
        return self._data_infos
    
    def __repr__(self):
        repr_str = f"Dataset in: {self.path}\n"
        return repr_str