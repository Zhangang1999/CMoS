import logging
from collections import OrderedDict
from typing import Dict, List

import numpy as np
from metrics import METRICS
from utils.instantiate import instantiate_from_args
from utils.time_utils import LogTimer
from utils.metric_utils import calc_metric

from hooks import HOOKS, BaseHook


@HOOKS.register()
class MetricHook(BaseHook):

    def __init__(self, metric_cfg) -> None:
        super().__init__()
        self.metric_cfg = metric_cfg
        self.best_metric = -1
        self.is_ascending = metric_cfg.is_ascending

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

        self._update_best_metric(metrics, trainer)

        trainer.file_manager.log_log(
            session='status',
            data=dict(msg="Summaring metrics.")
        )

    def _update_best_metric(self, metrics, trainer):
        
        curr_metric = metrics['all'][trainer.metric_labels[0]]
        if (self.best_metric > curr_metric and self.is_ascending) \
            or (self.best_metric < curr_metric and not self.is_ascending):
            return

        self.best_metric = curr_metric
        trainer.save_checkpoint('best')

        trainer.file_manager.log_log(
            session='status',
            data=dict(
                msg="Save checkpoint best.",
                key=trainer.metric_labels[0],
                value=self.best_metric
            )
        )

    def _collect_metrics(self, metric_classes:List[str]) -> Dict:
        """Collecting the metrics to dict."""
        return calc_metric(self.metric_dict, metric_classes)
