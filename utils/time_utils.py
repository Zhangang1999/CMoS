
import datetime
import logging
import time

def get_time_str()->str:
    ''' function: get the time

        return: str
        example: 2022-01-07_10:59:03
    '''
    return time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

def get_eta_str(max_iter:int , iteration:int, time_avg:float)->str:
    ''' function: compute the rest of the time  by (max_iter-iteration) * time_avg
        params:
                 max_iter : the max iteration
                 iteration: current iteration
                 time_avg : the used average time per round
        return:  the rest time like H:MM:SS (0:13:52)

    '''
    eta_str = str(datetime.timedelta(
        seconds=(max_iter-iteration) * time_avg.get_avg())).split('.')[0]
    return eta_str

def get_tot_str(start, end):
    tot_str = str(datetime.timedelta(
        seconds=(end - start))).split('.')[0]
    return tot_str

class LogTimer:

    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        self.start_time = time.time()
        logging.info(self.msg + ' ...')

    def __exit__(self, e, ev, t):
        self.end_time = time.time() 
        logging.info(self.msg + 'done. using {:.4}s'.format(
            self.start_time-self.end_time))  
