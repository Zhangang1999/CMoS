
from .base_metric import METRICS, BaseMetric
import numpy as np

@METRICS.register()
class Acc(BaseMetric):

    def __init__(self, metric: str, num_classes):
        super().__init__(metric)
        self.num_classes = num_classes

    def __call__(self, x, y, name=None):
        if name is None:
            name = str(self.name_cache)
            self.name_cache += 1

        self.ResDict.update({name: {}})
        for _cls in range(1, self.num_classes):
            p = (x == _cls).astype(np.bool).reshape(-1)
            t = (y == _cls).astype(np.bool).reshape(-1)
            acc = self._calc_acc(p, t)
            self.ResDict[name][str(_cls)] = acc

    @staticmethod
    def _calc_acc(x, y):
        tp = np.count_nonzero(x & y)
        # fp = np.count_nonzero((not x) & y)
        # tn = np.count_nonzero((not x) & (not y))
        # fn = np.count_nonzero(x & (not y))
        return tp / x.shape[0]