import numpy

class AbstractLayer:
    def __init__(self, optimizer = None):
        self._optimizer = None
    def forward(self, x):
        pass
    def backward(self, x, derr):
        pass
    def batch_forward(self, x):
        pass
    def batch_backward(self, x, derr):
        pass
    def epoch_end(self):
        pass
