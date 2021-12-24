
from metrics import METRICS, BaseMetric
import numpy as np

@METRICS.register()
class ACC(BaseMetric):

    def __init__(self, metric: str, num_classes):
        super().__init__(metric)
        self.num_classes = num_classes

    def __call__(self, x, y, name=None):
        if name is None:
            name = str(self.name_cache)
            self.name_cache += 1

        self.ResDict.update({name: {}})
        for _cls in range(self.num_classes):
            p = (x == _cls).astype(bool).reshape(-1)
            t = (y == _cls).astype(bool).reshape(-1)
            conf_matrix = self._calc_tp_fp_tn_fn(p, t)
            self.ResDict[name][_cls] = conf_matrix
        # self.ResDict[name]['acc'] = [acc]

    @staticmethod
    def _calc_tp_fp_tn_fn(x, y):
        tp = np.count_nonzero(x * y)
        fp = np.count_nonzero(x * (~y))
        tn = np.count_nonzero((~x) * (~y))
        fn = np.count_nonzero((~x) * y)
        return [tp, fp, tn, fn]