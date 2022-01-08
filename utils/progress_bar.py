import time


class ProgressBar():
    """ function: A simple progress bar that just outputs a string.
        params:
                length  : set the length of progress bar string default:200
                max_val : the max of the progress
        example:
        '''
                progress_bar = ProgressBar(length=200, max_val=100) #init
                for i in range(progress_bar.max_val):
                    time.sleep(1)
                    # https://www.cnblogs.com/zzliu/p/10156658.html
                    print('\rProcessing:  %s %6d / %6d        '
                          % (repr(progress_bar), i+1, progress_bar.max_val), end='') # /r return to the beginning,end="" delete the \n
                    progress_bar.set_val(i+1)
        '''
    """

    def __init__(self, length:int=150, max_val:int=100):
        self.max_val = max_val
        self.length = length
        self.cur_val = 0
        
        self.cur_num_bars = -1
        self._update_str()

    def set_val(self, new_val):
        self.cur_val = new_val

        if self.cur_val > self.max_val:
            self.cur_val = self.max_val
        if self.cur_val < 0:
            self.cur_val = 0

        self._update_str()
    
    def is_finished(self):
        return self.cur_val == self.max_val

    def _update_str(self):
        num_bars = int(self.length * (self.cur_val / self.max_val))

        if num_bars != self.cur_num_bars:
            self.cur_num_bars = num_bars
            self.string = '█' * num_bars + '░' * (self.length - num_bars)
    
    def __repr__(self):
        return self.string
    
    def __str__(self):
        return self.string