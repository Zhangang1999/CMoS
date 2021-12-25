
import sys
import os
import json
import unittest

cur_path = os.path.dirname(os.path.abspath(__file__).replace('\\', '/'))
sys.path.insert(0, f'{cur_path}/../')

from managers import FileManager

class FileManagerTest(unittest.TestCase):

    def setUp(self) -> None:
        self.mgr = FileManager('./')
        self.mgr.makedirs('test')
    
    def test_path(self):
        self.assertEqual(os.path.basename(self.mgr.root), 'CMoS')
        self.assertEqual(self.mgr.ckpt('test'), os.path.abspath('./assets/runs/test/weights'))
    
    def test_logs(self):
        self.mgr.log_init('test', {'header': 'test_header'})

    def test_file(self):
        self.assertIsInstance(self.mgr.ckpts('test'), list)
        self.assertEqual(len(self.mgr.logs('test')), 3)

    @unittest.expectedFailure
    def test_log_fail(self):
        self.mgr.log_log('test', 'fail', {"test_content": "sth. like that."})

    def test_log_content(self):
        self.mgr.log_log('test', 'train', {"test_content": "sth. like that."})

        with open(self.mgr.logs('test')['train'], 'r') as f:
            contents = f.readlines()
        self.assertEqual(json.loads(contents[1])['data'], {"test_content": "sth. like that."})

if __name__ == "__main__":
    unittest.main()