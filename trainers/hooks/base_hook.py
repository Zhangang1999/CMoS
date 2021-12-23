
from managers.ops_manager import OpsManager
HOOKS = OpsManager('hook')

class BaseHook(object):
    STAGES = [
        'before_run', 'after_run',
        'before_train_epoch', 'after_train_epoch',
        'before_valid_epoch', 'after_valid_epoch',
        'before_train_iter', 'after_valid_iter',
        'before_valid_iter', 'after_valid_iter',
    ]

    def __init__(self) -> None:
        super().__init__()

        self._priority = 0
    
    @property
    def priority(self):
        return self._priority
    @priority.setter
    def set_priority(self, priority):
        self._priority  = priority
    
    def before_run(self, trainer):
        raise NotImplementedError

    def after_run(self, trainer):
        raise NotImplementedError

    def before_epoch(self, trainer):
        raise NotImplementedError

    def after_epoch(self, trainer):
        raise NotImplementedError

    def before_iter(self, trainer):
        raise NotImplementedError

    def after_iter(self, trainer):
        raise NotImplementedError

    def before_train_epoch(self, trainer):
        self.before_epoch(trainer)

    def after_train_epoch(self, trainer):
        self.after_epoch(trainer)

    def before_valid_epoch(self, trainer):
        self.before_epoch(trainer)

    def after_valid_epoch(self, trainer):
        self.after_epoch(trainer)

    def before_train_iter(self, trainer):
        self.before_iter(trainer)

    def after_train_iter(self, trainer):
        self.after_iter(trainer)

    def before_valid_iter(self, trainer):
        self.before_iter(trainer)

    def after_valid_iter(self, trainer):
        self.after_iter(trainer)

    def every_n_epochs(self, trainer, n):
        return (trainer.epoch + 1) % n == 0 if n > 0 else False

    def every_n_inner_iters(self, trainer, n):
        return (trainer.inner_iter + 1) % n == 0 if n > 0 else False

    def every_n_iters(self, trainer, n):
        return (trainer.iter + 1) % n == 0 if n > 0 else False

    def is_last_epoch(self, trainer):
        return trainer.epoch + 1 == trainer._max_epochs

    def is_last_iter(self, trainer):
        return trainer.iter + 1 == trainer._max_iters

    def _check_method_or_stage_if_overridden(self, method_or_stage):
        return getattr(BaseHook, method_or_stage) != getattr(self, method_or_stage)

    def get_triggered_stages(self):
        trigger_stages = set()
        for stage in BaseHook.STAGES:
            if self._check_method_or_stage_if_overridden(stage):
                trigger_stages.add(stage)

        method_stages_map = dict(
            before_epoch=['before_train_epoch', 'before_valid_epoch'],
            after_epoch=['after_train_epoch', 'after_valid_epoch'],
            before_iter=['before_train_iter', 'before_valid_iter'],
            after_iter=['after_train_iter', 'after_valid_iter']
        )

        for method, map_stages in method_stages_map.items():
            if self._check_method_or_stage_if_overridden(method):
                trigger_stages.update(map_stages)
        
        return [stage for stage in BaseHook.STAGES if stage in trigger_stages]
