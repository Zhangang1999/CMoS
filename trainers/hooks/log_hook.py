
import time
from typing import List

from utils.moving_average import MovingAverage
from utils.time_utils import get_eta_str, get_tot_str
from utils.metric_utils import format_metric_msg

from .base_hook import BaseHook
from .hook_builder import HOOKS


@HOOKS.register()
class LogHook(BaseHook):

    def __init__(self, 
                 max_window_size=100, 
                 print_loss_freq=50,
                 ) -> None:
        super().__init__()
        self.max_window_size = max_window_size
        self.print_loss_freq = print_loss_freq

    def before_run(self, trainer):
        """Init the moving averages and record the time."""

        self.loss_avgs = {
            k: MovingAverage(self.max_window_size)
            for k in trainer.loss_labels
        } 
        self.time_avgs = MovingAverage()
        self.start_time = time.time()

    def after_run(self, trainer):
        """record the time and print the time message."""
        
        self.end_time = time.time()
        time_msg = "Total training time: {}".format(get_tot_str(self.start_time, self.end_time))
        print(time_msg, flush=True)

        trainer.file_manager.log_log(
            session='train',
            data={'msg': time_msg}
        )

    def after_train_iter(self, trainer):
        """record the time and print the loss message."""

        if hasattr(self, 'cur_time'):
            self.cur_elapsed = time.time() - self.cur_time
            self.time_avgs.add(self.cur_elapsed)
        self.cur_time = time.time()
        self.cur_elapsed = 0.

        loss_labels = trainer.loss_labels
        losses = trainer.outputs['losses']
        for k in loss_labels:
            self.loss_avgs[k].add(losses[k].item())

        if self.every_n_iters(trainer, self.print_loss_freq):
            msg = self._format_loss_msg(trainer, loss_labels)
            print(msg, flush=True)

        trainer.file_manager.log_log(
            session='train',
            data={'msg': msg}
        )

    def after_valid_epoch(self, trainer):
        """format and print the metric message."""

        assert hasattr(trainer, 'metric_classes')
        metric_labels = trainer.metric_labels
        metric_classes = trainer.metric_classes

        msg = self._format_metric_msg(trainer, metric_labels, metric_classes)
        print(msg, flush=True)

        trainer.file_manager.log_log(
            session='train',
            data={'msg': msg}
        )

    def _format_loss_msg(self, trainer, loss_labels:List[str]) -> str:
        """format the loss messages."""

        stage_msg = "[%3d] %5d ||" % (trainer.epoch, trainer.iter)

        losses = sum([[k, self.loss_avgs[k].get_avg()] for k in loss_labels], [])
        total_losses = sum([self.loss_avgs[k].get_avg() for k in loss_labels])
        loss_msg = (" %s: %.3f |" * len(loss_labels) + " T: %.3f ||") % tuple(losses + [total_losses])

        time_msg = " ETA: %s || timer: %.3f'" \
            % (get_eta_str(trainer.max_iters, trainer.iter, self.time_avgs), self.cur_elapsed)
        
        return stage_msg + loss_msg + time_msg

    def _format_metric_msg(self, trainer, metric_labels, metric_classes):
        """format the metric messages."""
        return format_metric_msg(trainer.outputs['metrics'], metric_labels, metric_classes)