
import math
from .base_hook import BaseHook
from .hook_builder import HOOKS

@HOOKS.register()
class LrHook(BaseHook):
    POLICY = [
        'cos',
        'step'
    ]

    def __init__(self, policy:str, extra_kwargs={}) -> None:
        super().__init__()
        assert policy in self.POLICY
        self.policy = policy
        self.extra_kwargs = extra_kwargs

        self.init_lr = []
        self.next_lr = []

    def before_run(self, trainer):
        """Record the init lr and get the necessary args."""

        for group in trainer.optimizer.param_groups:
            self.init_lr.append(group['lr'])

        if self.policy == 'step':
            self.step_idx = 0
            self.gamma = self.extra_kwargs['gamma']
            self.lr_steps = self.extra_kwargs['lr_steps']
        
            trainer.file_manager.log_log(
                session='debug',
                data=dict(
                    lr_policy=self.policy,
                    kwargs=self.extra_kwargs)
            )

    def after_train_iter(self, trainer):
        """get and set the next lr."""
        
        self.next_lr = self._get_next_lr(trainer)
        self._set_lr(trainer, self.next_lr)

        trainer.file_manager.log_log(
            session='debug',
            data={"lr": self.next_lr}
        )

    def _get_next_lr(self, trainer):
        """Get the next learning rate."""

        curr_lr = trainer.current_lr()

        if self.policy == 'cos':
            next_lr = [lr * 0.5 * (1. + math.cos(math.pi * trainer.iter / trainer.max_iters))
                        for lr in curr_lr]
        elif self.policy == 'step' and trainer.iter >= self.lr_steps[self.step_idx]:
            next_lr = [lr * (self.gamma**self.step_index) for lr in curr_lr] 

        return next_lr

    def _set_lr(self, trainer, lrs):
        """Set the learning rate."""
        for group, lr in zip(trainer.optimizer.param_groups, lrs):
            group['lr'] = lr