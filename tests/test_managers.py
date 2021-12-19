
import sys
import os

cur_path = os.path.dirname(os.path.abspath(__file__).replace('\\', '/'))
sys.path.insert(0, f'{cur_path}/../')

from managers.file_manager import FileManager
from managers.log_manager import LogManager
from managers.path_manager import PathManager

path_m = PathManager('./')
file_m = FileManager(path_m)
logger = LogManager(file_m)

logger.init_log('test', {'tt':123})
logger.log('test', 'train', {'ll':11})
logger.log('test', 'debug', {'ll':11})
logger.log('test', 'status', {'ll':11})