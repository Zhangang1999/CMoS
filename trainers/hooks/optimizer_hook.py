import logging
from .base_hook import HOOKS, BaseHook
from torch.nn.utils import clip_grad

@HOOKS.register()
class OptimizerHook(BaseHook):

    def __init__(self, grad_clip=None, max_loss=2000.0) -> None:
        super().__init__()

        self.grad_clip = grad_clip
        self.max_loss = max_loss

    def after_train_iter(self, trainer):
        losses = trainer.outputs['losses']
        trainer.optimizer.zero_grad()
        if torch.isnan(losses) or losses > self.max_loss:
            #TODO: Add log here.
            return
        losses.backward()
        if self.grad_clip is not None:
            grad_norm = self._clip_grads(trainer.model.parameters())
            #TODO: Add log here.
        trainer.optimizer.step()

    def _clip_grads(self, params):
        params = list(filter(
            lambda p: p.requires_grad and p.grad is not None, params
        ))
        if len(params) > 0:
            return clip_grad.clip_grad_norm(params, **self.grad_clip)
