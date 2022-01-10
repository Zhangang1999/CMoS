import json
import logging

from torch.nn.utils import clip_grad

from .base_hook import BaseHook
from .hook_builder import HOOKS


@HOOKS.register()
class OptimizerHook(BaseHook):

    def __init__(self, grad_clip=None, max_loss=2000.0) -> None:
        super().__init__()
        self.grad_clip = grad_clip
        self.max_loss = max_loss

    def after_train_iter(self, trainer):
        """check the loss/grad if valid and update."""

        losses = trainer.outputs['losses']
        trainer.optimizer.zero_grad()
        
        if torch.isnan(losses) or losses > self.max_loss:
            msg = "Loss outbounds."
            logging.info(msg)
            trainer.file_manager.log_log(
                session='debug',
                data=dict(msg=msg)
            )
            return

        losses.backward()
        if self.grad_clip is not None:
            grad_norm = self._clip_grads(trainer.model.parameters())
            
            trainer.file_manager.log_log(
                session='debug',
                data=dict(grad_norm=grad_norm)
            )

        trainer.optimizer.step()

    def _clip_grads(self, params):
        """clipping the gradients."""

        params = list(filter(
            lambda p: p.requires_grad and p.grad is not None, params
        ))

        if len(params) > 0:
            return clip_grad.clip_grad_norm(params, **self.grad_clip)
