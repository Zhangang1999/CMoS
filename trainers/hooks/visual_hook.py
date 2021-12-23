
from .base_hook import HOOKS, BaseHook

@HOOKS.register()
class VisualHook(BaseHook):

    def __init__(self, visualizer) -> None:
        super().__init__()
        self.visualizer = visualizer

    def after_valid_epoch(self, trainer):
        valid_loss = trainer.outputs['losses']
        self.visualizer.add_scalar("valid_loss", valid_loss, trainer.epoch)

    def after_train_iter(self, trainer):
        train_loss = trainer.outputs['losses']
        self.visualizer.add_scalar("train_loss", train_loss, trainer.iter)