import numpy as np
from NeuralNetwork.Layer import AbstractLayer
import math
import time

#Activation Function
af_sigmoid =  lambda x : 1/(1+np.exp(-x))
af_relu = np.vectorize(lambda x : x if x>0 else 0)
daf_relu = np.vectorize(lambda x : 1 if x>0 else 0)
af_lrelu = np.vectorize(lambda x : x if x>0 else 0.0001 * x)
daf_lrelu = np.vectorize(lambda x : 1 if x>0 else 0.0001)
af_selu = np.vectorize(lambda x : 1.0507 * (1.67326 * (math.exp(x)-1) if x <0 else x))
daf_selu = np.vectorize(lambda x :  1.0507 * (1.67326 * (math.exp(x)) if x <0 else 1))
af_softplus = np.vectorize(lambda x : math.log(1+math.exp(x)))
daf_softplus = np.vectorize(lambda x : af_sigmoid(x))
af_sin = np.vectorize(lambda x : math.sin(x))
daf_sin = np.vectorize(lambda x : math.cos(x))
af_linear = np.vectorize(lambda x : x)
daf_linear = np.vectorize(lambda x : 1)

def af_tanh(x):
    ex = np.exp(x)
    emx = np.exp(-x)
    return (ex - emx)/(ex + emx)

def daf_tanh(x):
    tanh = af_tanh(x)
    return 1-tanh**2

def daf_sigmoid(x):
    sigmoid = af_sigmoid(x)
    return sigmoid * (1-sigmoid)


def af_softmax(x):
    shiftx = x - np.max(x) + 1e-100
    exps = np.exp(shiftx) 
    return exps / np.sum(exps)


def daf_softmax(x):
    s = af_softmax(x)
    return np.multiply(s,(1-s))

class AbstractActivationLayer(AbstractLayer):
    def __init__(self, af=None, daf=None):
        self.__af = af
        self.__daf = daf

    def forward(self, X, out=None):
        return self.__af(X)

    def backward(self, X, derr):
        return np.multiply(self.__daf(X), derr)

    def batch_forward(self, X, out=None):
        return np.apply_along_axis(self.forward, 1, X, out=out)

    def batch_backward(self, X, derr, out=None):
        return np.multiply(np.apply_along_axis(self.__daf, 1, X), derr, out=out)


class Relu(AbstractActivationLayer):
    def __init__(self):
        super().__init__(af_relu, daf_relu)

class Sigmoid(AbstractActivationLayer):
    def __init__(self):
        super().__init__(af_sigmoid, daf_sigmoid)

class Softmax(AbstractActivationLayer):
    def __init__(self):
        super().__init__(af_softmax, daf_softmax)

class LeakyRelu(AbstractActivationLayer):
    def __init__(self):
        super().__init__(af_lrelu, daf_lrelu)

class Selu(AbstractActivationLayer):
    def __init__(self):
        super().__init__(af_selu, daf_selu)

class SoftPlus(AbstractActivationLayer):
    def __init__(self):
        super().__init__(af_softplus, daf_sodtplus)

class Sinusoid(AbstractActivationLayer):
    def __init__(self):
        super().__init__(af_sin, daf_sin) 

class Linear(AbstractActivationLayer):
    def __init__(self):
        super().__init__(af_linear, daf_linear)

class CrossEntropySoftmax(AbstractActivationLayer):
    def __init__(self):
        super().__init__(af_softmax, daf_linear)

    def batch_backward(self, X, derr, out=None):
        return derr

class Tanh(AbstractActivationLayer):
    def __init__(self):
        super().__init__(af_tanh, daf_tanh)
