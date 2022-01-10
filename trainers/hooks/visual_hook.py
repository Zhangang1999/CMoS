
from .base_hook import BaseHook
from .hook_builder import HOOKS


@HOOKS.register()
class VisualHook(BaseHook):

    def __init__(self, visualizer) -> None:
        super().__init__()
        self.visualizer = visualizer

    def after_valid_epoch(self, trainer):
        """Add info to visualizer."""

        valid_loss = trainer.outputs['losses']
        self.visualizer.add_scalar("valid_loss", valid_loss, trainer.epoch)

        trainer.file_manager.log_log(
            session='status',
            data=dict(msg="Adding Info to the visualizer.")
        )

    def after_train_iter(self, trainer):
        """Add info to visualizer."""

        train_loss = trainer.outputs['losses']
        self.visualizer.add_scalar("train_loss", train_loss, trainer.iter)

        trainer.file_manager.log_log(
            session='status',
            data=dict(msg="Adding Info to the visualizer.")
        )
