import unittest

import os
import time
import sys
cur_path = os.path.dirname(os.path.abspath(__file__).replace('\\', '/'))
sys.path.insert(0, f'{cur_path}/../')

from trainers.trainer_builder import TRAINERS
from trainers.hooks.hook_builder import HOOKS
from managers import FileManager

class FakeLoss(object):
    def __init__(self, value) -> None:
        self._value = value
    
    def item(self):
        return self._value

class HookTest(unittest.TestCase):

    def setUp(self) -> None:
        self.trainer = TRAINERS.get('EpochTrainer')(
            model=None,
            optimizer=None,
            file_manager=FileManager('./'),
            train_params={'max_epochs':1, 'max_iters':1}
        )

        self.trainer.register_hook(
            hook=HOOKS.get('LogHook')(
                max_window_size=100,
                print_loss_freq=1
            ),
            priority=100,
        )
        self.trainer.loss_labels = ['C', 'Z', 'J']
        self.trainer.metric_labels = ['acc']
        self.trainer.metric_classes = ['A', 'B']

    def test_hook_info(self):
        print(self.trainer.get_hook_info())

    def test_call_hook_before_run(self):
        self.trainer.call_hook("before_run")
        hook = self.trainer.hooks[0]
        self.assertEquals(len(hook.loss_avgs), 3)

    def test_call_hook_after_run(self):
        self.trainer.call_hook("before_run")
        time.sleep(2)
        self.trainer.call_hook("after_run")
    
    def test_call_hook_after_iter(self):
        self.trainer.call_hook("before_run")
        setattr(self.trainer, 'outputs', {})
        self.trainer.outputs['losses'] = {
            'C': FakeLoss(1.11), 
            'Z': FakeLoss(2.22), 
            'J': FakeLoss(3.33)
        }
        self.trainer.call_hook("after_train_iter")

    def test_call_hook_after_valid_epoch(self):
        self.trainer.call_hook("before_run")
        setattr(self.trainer, 'outputs', {})
        self.trainer.outputs['metrics'] = {
            'acc': {
                'all': 2.22,
                'A': 1.11,
                'B': 3.33,
            }
        }
        self.trainer.call_hook("after_valid_epoch")


if __name__ == "__main__":
    unittest.main()