
from typing import Dict
from addict import Dict as ADDict

class Config(ADDict):
    """translate dict config to Config class. """
    def __init__(self, cfg:Dict) -> None:
        assert isinstance(cfg, Dict), f"cfg should be dict, but got {type(cfg)}"
        
        super().__init__(self, cfg)
