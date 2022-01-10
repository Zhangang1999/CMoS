
from .base_hook import BaseHook
from .hook_builder import HOOKS

@HOOKS.register()
class ModelHook(BaseHook):

    def __init__(self, save_ckpt_freq) -> None:
        super().__init__()
        self.save_ckpt_freq = save_ckpt_freq
        
    def before_run(self, trainer):
        """Record the status and log."""

        trainer.file_manager.log_log(
            session='status',
            data={'msg': "Start Training."}
        )

    def after_valid_epoch(self, trainer):
        """Save the checkpoint and log."""
        
        if not self.every_n_epochs(self.save_ckpt_freq):
            return

        trainer.save_checkpoint('latest')
        trainer.file_mangaer.log_log(
            session='status',
            data={'msg': "Save checkpoint latest."}
        )

    def after_run(self, trainer):
        """Save the checkpoint and log."""       

        trainer.save_checkpoint('latest')
        trainer.file_mangaer.log_log(
            session='status',
            data={'msg': "Train end. Save checkpoint latest."}
        )
