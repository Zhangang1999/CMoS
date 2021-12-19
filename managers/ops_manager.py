
class OpsManager(object):

    def __init__(self, name:str) -> None:
        super().__init__()
        self._name = name
        self._ops_dict = {}

    @property
    def name(self):
        return self._name

    @property
    def ops_dict(self):
        return self._ops_dict

    def __len__(self):
        return len(self._ops_dict)

    def __repr__(self) -> str:
        repr_str = f"Class name: {self.__class__.__name__}\n"
        repr_str += f"Register name: {self.name}\n"
        repr_str += f"Ops dict: {self.ops_dict}"
        return repr_str

    def get(self, name):
        return self._ops_dict.get(name, None)
    
    def __contains__(self, item):
        return self.get(item) is not None
    
    def _do_register(self, obj, obj_name:str, override=False):
        if not override and obj_name in self.ops_dict:
            raise ValueError(f"{obj_name} has already registered in Pool. \
                                keys:{list(self._ops_dict.keys())}")
        self._ops_dict[obj_name] = obj

    def deregister(self, obj_name):
        if obj_name in self._ops_dict:
            self._ops_dict.pop(obj_name)
    
    def _check_register_params_if_valid(self, obj_name, override):
        if (obj_name is not None) and (not isinstance(obj_name, str)):
            raise ValueError("obj_name should be str")
        
        if not isinstance(override, bool):
            raise ValueError("override should be bool")
        
    def register(self, obj=None, obj_name=None, override=False):
        self._check_register_params_if_valid(obj_name, override)

        if obj is not None:
            obj_name = obj.__name__ if obj_name is not None else obj_name
            self._do_register(obj, obj_name, override)
            return obj
        else:
            def _register(_obj):
                obj_name = _obj.__name__
                self._do_register(_obj, obj_name, override)
                return _obj     
            return _register

