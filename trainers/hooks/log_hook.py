
from time import strftime, time
from .base_hook import BaseHook, HOOKS
from utils.time_utils import get_eta_str
from utils.moving_average import MovingAverage

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
        self.loss_avgs = {
            k: MovingAverage(self.max_window_size)
            for k in trainer.loss_labels
        } 
        self.time_avgs = MovingAverage()
        self.start_time = time.time()
    
    def after_run(self, trainer):
        self.end_time = time.time()
        time_msg = "Total training time: {}".format(
            (self.end_time-self.start_time).strftime("%H:%M:%S"))
        print(time_msg, flush=True)

    def after_train_iter(self, trainer):
        if hasattr(self, 'cur_time'):
            self.cur_elapsed = time.time() - self.cur_time
            self.time_avgs.add(self.cur_elapsed)
        self.cur_time = time.time()

        loss_labels = trainer.model.loss.loss_labels
        losses = trainer.outputs['losses']
        for k in loss_labels:
            self.loss_avgs[k].add(losses[k].item())

        if self.every_n_iters(trainer, self.print_loss_freq):
            msg = self._format_loss_msg(trainer, loss_labels)
            print(msg, flush=True)
        
    def after_valid_epoch(self, trainer):
        assert hasattr(trainer, 'metric_labels')
        assert hasattr(trainer, 'metric_classes')
        metric_labels = trainer.metric_labels
        metric_classes = trainer.metric_classes

        msg = self._format_metric_msg(trainer, metric_labels, metric_classes)
        print(msg, flush=True)

    def _format_loss_msg(self, trainer, loss_labels):
        stage_msg = "[%3d] %5d ||" % (trainer.epoch, trainer.iter)

        losses = sum([[k, self.loss_avgs[k].get_avg()] for k in loss_labels], [])
        total_losses = sum([self.loss_avgs[k].get_avg() for k in loss_labels])
        loss_msg = " %s: %.3f |" * len(loss_labels) + " T: %.3f ||" % tuple(losses + total_losses)

        time_msg = " ETA: %s || timer: %.3f'" \
            % (get_eta_str(trainer.max_iters, trainer.iter, self.time_avgs), self.cur_elapsed)
        
        return stage_msg + loss_msg + time_msg

    def _format_metric_msg(self, trainer, metric_labels, metric_classes):
        make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
        make_sep = lambda n: ('-------+' * n)

        metrics = trainer.outputs['metrics']

        title_msg = '\n'
        title_msg += make_row([''] + [(' '+ x + ' ') for x in metric_classes]) + '\n'
        title_msg += make_sep(len(metric_classes)+1)

        metric_msg = '\n'
        for metric in metric_labels:
            metric_msg += make_row([metric] + ['%.2f' % x if x<100 else '%.1f' \
                            % x for x in metrics[metric].values()])
            metric_msg += '\n'

        ending_msg = make_sep(len(metric_labels)+1) + '\n'
        ending_msg += '\n'

        return title_msg + metric_msg + ending_msg
    