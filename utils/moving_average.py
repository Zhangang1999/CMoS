import math
from collections import deque

""" https://github.com/dbolya/yolact/tree/master/utils """
class MovingAverage():
    """ Keeps an average window of the specified number of items. """

    def __init__(self, max_window_size:int=1000):
        self.max_window_size = max_window_size
        self.reset()

    def add(self, elem):
        """ Adds an element to the window, removing the earliest element if necessary. """
        if not math.isfinite(elem):
            print('Warning: Moving average ignored a value of %f' % elem)
            return
        
        self.window.append(elem)
        self.sum += elem

        if len(self.window) > self.max_window_size:
            self.sum -= self.window.popleft()

    def append(self, elem):
        self.add(elem)

    def reset(self):
        """ Resets the MovingAverage to its initial state. """
        self.window = deque()
        self.sum = 0

    def get_avg(self):
        """ Returns the average of the elements in the window. """
        return self.sum / max(len(self.window), 1)

    def __str__(self):
        return str(self.get_avg())
    
    def __repr__(self):
        return repr(self.get_avg())
    
    def __len__(self):
        return len(self.window)
if __name__ == "__main__":
    movingaverage = MovingAverage()
