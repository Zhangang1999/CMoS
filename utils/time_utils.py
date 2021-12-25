
import datetime
import logging
import time

def get_time_str():
    return time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

def get_eta_str(max_iter, iteration, time_avg):
    eta_str = str(datetime.timedelta(
        seconds=(max_iter-iteration) * time_avg)).split('.')[0]
    return eta_str

class LogTimer:

    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        self.start_time = time.time()
        logging.log(self.msg + ' ...')

    def __exit__(self, e, ev, t):
        self.end_time = time.time() 
        logging.log(self.msg + 'done. using {:.4}s'.format(
            self.start_time-self.end_time))  