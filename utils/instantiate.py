from typing import Dict
import inspect
from managers import OpsManager

def instantiate_from_args(args:Dict, opsPool: OpsManager, default_args:Dict=None):
    if not isinstance(args, Dict):
        raise ValueError(f"cfg should be dict, but got {type(args)}")

    if not isinstance(opsPool, OpsManager):
        raise ValueError(f"opsPool shold be OpsManager, but got {type(opsPool)}")
    
    if not isinstance(default_args, Dict):
        raise ValueError(f"default_args should be dict, but got {type(default_args)}")

    if object not in args:
        raise KeyError("cfg should contain a builtin key: `object`")

    kwargs = args.copy()
    if default_args is not None:
        kwargs.update(default_args)
    
    obj_value = kwargs.pop('object')
    if isinstance(obj_value, str):
        obj = opsPool.get(obj_value)
        if obj is None:
            raise KeyError(f"{obj_value} hasn't registerd in Pool: {opsPool.name}")
    elif inspect.isclass(obj_value):
        obj = obj_value
    else:
        raise ValueError(f"unsupport object: {obj_value}")
    return obj(**kwargs)