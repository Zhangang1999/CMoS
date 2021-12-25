import logging
from collections import OrderedDict
from typing import Dict, List

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
        """Init some arguments."""

        trainer.metric_labels = self.metric_cfg.metric_labels
        trainer.metric_classes = self.metric_cfg.metric_classes

        trainer.file_manager.log_log(
            session='debug',
            data=dict(
                metric_labels=trainer.metric_labels,
                metric_classes=trainer.metric_classes
            )
        )

    def before_valid_epoch(self, trainer):
        """Instantiate metric objects."""

        with LogTimer("Instantiate metric objects"):
            self.metric_dict = {}
            for metric, obj_cfg in self.metric_cfg.to_eval_metrics.items():
                self.metric_dict.update(
                    {metric: instantiate_from_args(obj_cfg, METRICS)})

        msg = 'Instantiated Metrics: {}'.format(list(self.metric_dict.keys()))
        logging.info(msg)

        trainer.file_manager.log_log(
            session='status',
            data=dict(msg=msg)
        )
        
    def after_valid_iter(self, trainer):
        """Calculate metrics."""

        with LogTimer("Calculate metrics"):
            predicts = trainer.outputs['predicts']
            gts = trainer.outputs['gts']
            for _, metric_obj in self.metric_dict.items():
                metric_obj.__call__(predicts, gts)

        trainer.file_manager.log_log(
            session='status',
            data=dict(msg="Calculating metrics.")
        )

    def after_valid_epoch(self, trainer):
        """Summary the metrics."""

        with LogTimer("Summary metrics"):
            metrics = self._collect_metrics(trainer.metric_classes)
            trainer.outputs.update(metrics=metrics)

        trainer.file_manager.log_log(
            session='status',
            data=dict(msg="Summaring metrics.")
        )

    def _collect_metrics(self, metric_classes:List[str]) -> Dict:
        """Collecting the metrics to dict."""

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
