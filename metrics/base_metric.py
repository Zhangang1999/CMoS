
from typing import Dict, List
import typing

from numpy.lib.arraysetops import isin

class BaseMetric(object):
    
    def __init__(self, metric:str):
        self.metric = metric
        self.name_cache = 0
        self.__setattr__(metric+'Dict', {})

    @property
    def ResDict(self):
        return getattr(self, self.metric+'Dict')
    
    def __call__(self, x, y):
        raise NotImplementedError

    def gather(self, target_keys:List[str]) -> Dict:        
        """Gather the keys from ResDict of the metric.

        Args:
            target_keys (List[str]): List of keys to collect.

        Returns:
            Dict: Gathered Dict with target keys.
        """
        assert isinstance(target_keys, (list, tuple))
        def _collect(d, t):
            if not isinstance(d, dict): return []
            if t in d and isinstance(d[t], (list, float)):
                return d[t]

            s = []
            for k in d:
                s.append(_collect(d[k], t))
            return s

        gather_result = {}
        for target_key in target_keys:
            gather_result[target_key] = _collect(self.ResDict, target_key)

        return gather_result
