import logging
from collections import OrderedDict

import numpy as np
from metrics import METRICS
from utils.instantiate import instantiate_from_args
from utils.time_utils import LogTimer

from hooks import HOOKS, BaseHook


@HOOKS.register()
class MetricHook(BaseHook):

    def __init__(self, metric_cfg) -> None:
        super().__init__()
        self.metric_cfg = metric_cfg

    def before_run(self, trainer):
        trainer.metric_labels = self.metric_cfg.metric_labels
        trainer.metric_classes = self.metric_cfg.metric_classes

    def before_valid_epoch(self, trainer):
        with LogTimer("Instantiate metrics objects"):
            self.metric_dict = {}
            for metric, obj_cfg in self.metric_cfg.to_eval_metrics.items():
                self.metric_dict.update(
                    {metric: instantiate_from_args(obj_cfg, METRICS)})
        logging.info('Instantiated Metrics: {}'.format(list(self.metric_dict.keys())))
        
    def after_valid_iter(self, trainer):
        with LogTimer("Calculate metrics"):
            predicts = trainer.outputs['predicts']
            gts = trainer.outputs['gts']
            for _, metric_obj in self.metric_dict.items():
                metric_obj.__call__(predicts, gts)

    def after_valid_epoch(self, trainer):
        with LogTimer("Summary metrics"):
            metrics = self._collect_metrics(trainer.metric_classes)
            trainer.ouputs.update(metrics=metrics)

    def _collect_metrics(self, metric_classes):
        metrics = {}
        for metric, metric_obj in self.metric_dict.items():
            metrics.update({metric: OrderedDict()})
            metrics[metric]['all'] = 0
            resDict = metric_obj.gather(metric)

            for _cls in range(1, len(metric_classes)):
                mean = np.mean(np.array(resDict[_cls]))
                metrics[metric][metric_classes[_cls]] = mean
            metrics[metric]['all'] = \
                sum(metrics[metric].values()) / (len(metrics[metric].values())-1)
        return metrics