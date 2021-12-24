from typing import Dict
import numpy as np
import unittest

import os
import sys
cur_path = os.path.dirname(os.path.abspath(__file__).replace('\\', '/'))
sys.path.insert(0, f'{cur_path}/../')

from metrics import METRICS, ACC

class MetricTest(unittest.TestCase):

    def setUp(self) -> None:
        self.acc = METRICS.get('ACC')('acc', 2)

    def test_metrics(self):
        self.assertEqual(METRICS.name, 'metric')

    def test_instance(self):
        self.assertIsInstance(self.acc, ACC)

    def test_acc(self):
        y = np.array([1,0,1])

        for x in ([0,1,0], [1,0,1]):
            self.acc(np.array(x), y)
        
        self.assertEqual(self.acc.ResDict,
            {'0': {0: [0, 2, 0, 1], 1: [0, 1, 0, 2]}, 
             '1': {0: [1, 0, 2, 0], 1: [2, 0, 1, 0]}})

        gather_result = self.acc.gather([0, 1])
        self.assertEqual(gather_result, 
            {0: [[0, 2, 0, 1], [1, 0, 2, 0]], 
             1: [[0, 1, 0, 2], [2, 0, 1, 0]]})

if __name__ == '__main__':
    unittest.main()