
import math
from hooks import BaseHook, HOOKS

@HOOKS.register()
class LrHook(BaseHook):

    def __init__(self, policy, extra_kwargs={}) -> None:
        super().__init__()
        self.policy = policy
        self.extra_kwargs = extra_kwargs

        self._init_lr = []
        self._next_lr = []

    def before_run(self, trainer):
        for group in trainer.optimizer.param_groups:
            self._init_lr.append(group['lr'])

        if self.policy == 'step':
            self.step_idx = 0
            self.gamma = self.extra_kwargs['gamma']
            self.lr_steps = self.extra_kwargs['lr_steps']

    def before_train_iter(self, trainer):
        if trainer.iter == 0: 
            return
        self._next_lr = self._get_next_lr(trainer)
        self._set_lr(trainer, self._next_lr)
        #TODO: Add log here.

    def _get_next_lr(self, trainer):
        curr_lr = trainer.current_lr()

        if self.policy == 'cos':
            _next_lr = [lr * 0.5 * (1. + math.cos(math.pi * trainer.iter / trainer.max_iters))
                        for lr in curr_lr]
        
        if self.policy == 'step' and trainer.iter >= self.lr_steps[self.step_idx]:
            _next_lr = [lr * (self.gamma**self.step_index) for lr in curr_lr] 

        return _next_lr

    def _set_lr(self, trainer, lrs):
        for group, lr in zip(trainer.optimizer.param_groups, lrs):
            group['lr'] = lr