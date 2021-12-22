
import time

def get_time_str():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def get_eta_str(max_iter, iteration, time_avg):
    eta_str = str(datetime.timedelta(
        seconds=(max_iter-iteration) * time_avg)).split('.')[0]
    return eta_str