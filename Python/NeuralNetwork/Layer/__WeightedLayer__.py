from NeuralNetwork.Layer.__AbstractLayer__ import AbstractLayer
from NeuralNetwork.Optimizer import AdaMax, SGD, Momentum, Adam
from NeuralNetwork.Utils import convolve1d, dconvolve1d 
from NeuralNetwork.Layer.__ActivationFunction__ import Sigmoid, Tanh, Relu, LeakyRelu
#from numba import jit
import numpy as np
import copy
import time
from functools import partial


class Dense(AbstractLayer):
    def __init__(self, size, initializer=np.random.rand, optimizer = AdaMax, xavier=True, clip=None):
        #multiplier = np.sqrt(2/(np.prod(size))) if xavier else 1
        self.__W = np.array(initializer(size[0], size[1])) #* multiplier
        self.__B = np.array(initializer(1,size[1])) #*  multiplier
        self.__optimizer_W = optimizer(size)
        self.__optimizer_B = optimizer((size[1],))
        self.__clip = clip

    def forward(self, X, out=None):
        if out is not None:
            try:
                np.matmul(X, self.__W, out=out)
                np.add(out, self.__B, out=out)
                return out
            except Exception as e: 
                pass
        return X @ self.__W + self.__B


    def backward(self, X, derr):
        # X = X.reshape((X.shape[1], 1))
        self.__W-= self.__optimizer_W((X.T @ derr))
        self.__B-= self.__optimizer_B(derr)
        derr = derr @ self.__W.T 
        return derr

    def batch_forward(self, X, out=None):
        #return np.apply_along_axis(self.forward, 1, X)
        return self.forward(X, out)

    def batch_backward(self, X, derr, out=None):
        self.__W-= self.__optimizer_W((X.T @ derr))
        self.__B-= self.__optimizer_B(derr.sum(axis=0))
        if out is None:
            derr = derr @ self.__W.T 
            return derr
        else:
            np.matmul(derr, self.__W.T, out=out)
            return out


    def epoch_end(self):
        self.__optimizer_W.update()
        self.__optimizer_B.update()


class Convolve1d(AbstractLayer):
    def __init__(self, kernel_size, kernel_count, initializer=np.random.rand, optimizer=AdaMax, clip=None):
        self.__kernels = initializer(kernel_count, kernel_size[0], kernel_size[1])
        self.__optimizer_kernel = [optimizer((kernel_size[0], kernel_size[1])) for _ in range(kernel_count)]
        self.__clip = clip

    def __proto_backward(self, X, derr):
        dw, dx = zip(*[dconvolve1d(X, kernel, derr) for kernel in self.__kernels])
        dx = np.sum(dx, axis=0)
        return dx, dw

    def __proto_update(self, dw):
        dw = np.array([optimizer(d) for d, optimizer in zip(dw, self.__optimizer_kernel)])
        if self.__clip is not None:
           lower = self.__clip[0]
           upper = self.__clip[1]
           dw = dw.clip(lower, upper)
        self.__kernels -= dw

    def forward(self, X):
        #c = partial(convolve1d, X)
        #result = self.__pool.map(c, self.__kernels)
        #return np.sum(result, axis=0)
        '''ret = None
        for i, kernel in enumerate(self.__kernels):
            if ret is not None:
                ret += convolve1d(X, kernel)
            else:
                ret = convolve1d(X, kernel)
        return ret'''
        return np.sum(np.array([convolve1d(X, kernel) for kernel in self.__kernels]), axis=0)
        #return np.sum(c(self.__kernels), axis=0)

    def backward(self, X, derr):
        dx, dw = self.__proto_backward(self, X, derr)
        self.__proto_update(dw)
        return dx
   
    def batch_forward(self, X):
        return np.array([self.forward(x) for x in X])

    def batch_backward(self, X, derr):
        dx, dw = None, None
        for i in range(X.shape[0]):
           temp_dx, temp_dw = self.__proto_backward(X[i], derr[i])
           if i == 0:
              dw, dx = np.array(temp_dw), np.array(temp_dx)
           else:
              dw += temp_dw
              dx += temp_dx
        self.__proto_update(dw)
        return dx
        

class RNN(AbstractLayer):
    def __init__(self, size, s2v = False, initializer=np.random.rand, optimizer=AdaMax, clip=[-0.1,0.1], input_layer=None, hidden_layer=None):
        self.__input_layer = Dense((size[0], size[1]), initializer=initializer, optimizer=optimizer, clip=clip) if input_layer is None else input_layer
        self.__hidden_layer = Dense((size[1], size[1]), initializer=initializer, optimizer=optimizer, clip=clip) if hidden_layer is None else hidden_layer
        self.__size = size
        self.__s2v = s2v

    def __proto_s2v_forward(self, X):
        state = np.zeros((self.__size[1],)) + 1e-10
        sigmoid = Sigmoid()
        for x in X:
            state = self.__input_layer.forward(x) + self.__hidden_layer.forward(state)
            state = sigmoid.forward(state)
        return state

    def __proto_s2s_forward(self, X):
        state = np.zeros((self.__size[1],)) + 1e-10
        states = []
        sigmoid = Sigmoid()
        for x in X:
            state = self.__input_layer.forward(x) + self.__hidden_layer.forward(state)
            states.append(sigmoid.forward(state))
        return states
        
    def forward(self, X):
        if self.__s2v:
            return self.__proto_s2v_forward(X)
        else:
            return self.__proto_s2s_forward(X)

    def __proto_v2v_backward(self, X, derr, states):
        states = self.__proto_s2s_forward(X) if states is None else states
        ret_derr = [None] * len(states)
        sigmoid = Sigmoid()
        for i in range(len(states)-1, -1, -1):
            derr[i] = sigmoid.backward(states[i], derr[i])
            dx = self.__input_layer.backward(np.array([X[i]]), derr[i])
            dh = self.__hidden_layer.backward(states[i-1] if i>0 else np.zeros_like(states[i]), derr[i])
            derr[i-1] += dh if i != 0 else 0
            ret_derr[i] = dx
        return np.array(ret_derr)

    def __proto_s2v_backward(self, X, derr, states):
        states = self.__proto_s2s_forward(X) if states is None else states
        sigmoid = Sigmoid()
        ret_derr = [None] * len(states)
        for i in range(len(states)-1, -1, -1):
            derr = sigmoid.backward(states[i], derr) 
            dx = self.__input_layer.backward(np.array([X[i]]), derr)
            dh = self.__hidden_layer.backward(states[i-1] if i>0 else np.zeros_like(states[i]), derr)
            derr = dh
            ret_derr[i] = dx
        return ret_derr

    def backward(self, X, derr, states=None):
        if self.__s2v:
            return self.__proto_s2v_backward(X, derr, states)
        else:
            return self.__proto_v2v_backward(X, derr, states)

    def batch_forward(self, X):
        z = []
        for x in X:
            if not self.__s2v:
                z.append(self.forward(x))
            else:
                z = self.forward(x)
        return z

    def batch_backward(self, X, derr):
        sos = self.batch_forward(X)
        ret_derr = [None] * len(sos)
        for states, x, d in zip(sos, X, derr):
            self.backward(x, d, states)


class GRU(RNN):
    def __init__(self, size, s2v = False, initializer=np.random.rand, optimizer=AdaMax, clip=[-5,5]):
        self.__input_layer_r = Dense((size[0], size[1]), initializer=initializer, optimizer=optimizer, clip=clip)
        self.__hidden_layer_r = Dense((size[1], size[1]), initializer=initializer, optimizer=optimizer, clip=clip)
        self.__input_layer_u = Dense((size[0], size[1]), initializer=initializer, optimizer=optimizer, clip=clip)
        self.__hidden_layer_u = Dense((size[1], size[1]), initializer=initializer, optimizer=optimizer, clip=clip)
        self.__input_layer_c = Dense((size[0], size[1]), initializer=initializer, optimizer=optimizer, clip=clip)
        self.__hidden_layer_c = Dense((size[1], size[1]), initializer=initializer, optimizer=optimizer, clip=clip)
        self.__size = size
        self.__s2v = s2v
        self.__clip = clip

    def __proto_t_forward(self, x, state):
        sigmoid = Sigmoid()
        tanh = Tanh()
        r = sigmoid.forward(self.__input_layer_r.forward(x) + self.__hidden_layer_r.forward(state))
        u = sigmoid.forward(self.__input_layer_u.forward(x) + self.__hidden_layer_u.forward(state))
        c = tanh.forward(self.__input_layer_c.forward(x) + self.__hidden_layer_c.forward(state * r))
        state = u * state + (1-u) * c
        return r, u, c, state

    def __proto_t_backward(self, x, state, derr):
        sigmoid = Sigmoid()
        tanh = Tanh()
        p_r = self.__input_layer_r.forward(x) + self.__hidden_layer_r.forward(state)
        r = sigmoid.forward(p_r)
        p_u = self.__input_layer_u.forward(x) + self.__hidden_layer_u.forward(state)
        p_c = self.__input_layer_c.forward(x) + self.__hidden_layer_c.forward(state * r)
      

        derrdp_c = (1-sigmoid.forward(p_u)) * tanh.backward(p_c, derr) 
        derrdcdh = self.__hidden_layer_c.backward(state * r, derrdp_c) 
        derrdp_u = (state - tanh.forward(p_c)) * sigmoid.backward(p_u, derr) 
        derrdp_r = sigmoid.backward(p_r, derrdcdh * state)
        
 
        dx = self.__input_layer_u.backward(x, derrdp_u) + self.__input_layer_c.backward(x, derrdp_c) + self.__input_layer_r.backward(x, derrdp_r) 
        dh = self.__hidden_layer_u.backward(state, derrdp_u) + self.__hidden_layer_r.backward(state, derrdp_r) + derrdcdh * r
        l, u = self.__clip
        return dx.clip(l,u), dh.clip(l,u)


    def __proto_s2v_forward(self, X):
        state = np.zeros((self.__size[1],)) + 1e-10
        sigmoid = Sigmoid()
        tanh = Tanh()
        for x in X:
            _,_,_, state = self.__proto_t_forward(x, state)
        return state

    def __proto_s2s_forward(self, X):
        state = np.zeros((self.__size[1],)) + 1e-10
        states = []
        sigmoid = Sigmoid()
        for x in X:
            _,_,_, state = self.__proto_t_forward(x, state)
            states.append(state)
        return states
        
    def forward(self, X):
        if self.__s2v:
            return self.__proto_s2v_forward(X)
        else:
            return self.__proto_s2s_forward(X)

    def __proto_v2v_backward(self, X, derr, states):
        states = self.__proto_s2s_forward(X) if states is None else states
        ret_derr = [None] * len(states)
        for i in range(len(states)-1, -1, -1):
            dx, dh = self.__proto_t_backward(np.array([X[i]]), states[i-1] if i>0 else np.zeros_like(states[i])+ 1e-10, derr[i])
            derr[i-1] += dh
            ret_derr[i] = dx[0]
        return np.array(ret_derr)

    def __proto_s2v_backward(self, X, derr, states):
        states = self.__proto_s2s_forward(X) if states is None else states
        ret_derr = [None] * len(states)
        for i in range(len(states)-1, -1, -1):
            dx, dh = self.__proto_t_backward(np.array([X[i]]), states[i-1] if i>0 else np.zeros_like(states[i])+ 1e-10, derr)
            derr = dh if i > 0 else 0
            ret_derr[i] = dx[0]
        return ret_derr

    def backward(self, X, derr, states=None):
        if self.__s2v:
            return self.__proto_s2v_backward(X, derr, states)
        else:
            return self.__proto_v2v_backward(X, derr, states)

    def batch_forward(self, X):
        z = []
        for x in X:
            z.append(self.forward(x)[0])
        return np.array(z)

    def batch_backward(self, X, derr):
        ret_derr = [None] * len(X)
        for i, (x, d) in enumerate(zip(X, derr)):
            ret_derr[i] = self.backward(x, d, None)
        return ret_derr

class biGRU(AbstractLayer):
    def __init__(self, size, s2v = False, initializer=np.random.rand, optimizer=AdaMax, clip=[-5,5]):
        self.__forward_gru = GRU(size, s2v, initializer, optimizer, clip)
        self.__backward_gru = GRU(size, s2v, initializer, optimizer, clip)

    def forward(self, X): 
        a = self.__forward_gru.forward(X) + self.__backward_gru.forward(X.tolist()[::-1])
        return a

    def backward(self, X, derr):
        return self.__forward_gru.backward(X, derr) + self.__backward_gru.backward(X.tolist()[::-1], derr)

    def batch_forward(self, X):
        return np.array([self.forward(x) for x in X])

    def batch_backward(self, X, derr):
        return np.array([self.backward(x, d) for x, d in zip(X, derr)])

class FuzzyKohonen(AbstractLayer):
    def __init__(self, size, W=None, initializer=np.random.randn, alpha=0.1, alpha_decay=0.9):
        self.__W = W if W is not None else initializer(size[0], size[1])
        self.__alpha = alpha
        self.__alpha_decay = alpha_decay

    def __u(self, x):
        ds = [np.sum(np.power(v-x, 2)) for v in self.__W]
        u = [1/sum([np.power(di/dj, 2) for dj in ds]) for di in ds]
        return u

    def forward(self, X):
        ds = [np.sum(np.power(v-X, 2)) for v in self.__W]
        u = [1/sum([np.power(di/dj, 2) for dj in ds]) for di in ds]
        return np.argmax(u)

    def train(self, X):
        u = [self.__u(x) for x in X]
        old_W = copy.deepcopy(self.__W)
        for i in range(len(X)):
            for j in range(len(old_W)):
                self.__W[j] = self.__W[j] - self.__alpha * np.power(u[i][j],2) * (self.__W[j]-X[i])
        error = sum([np.sum(np.power(vk-vkm1, 2)) for vk, vkm1 in zip(self.__W, old_W)])
        return error
   
    def epoch_end(self):
        self.__alpha = self.__alpha * self.__alpha_decay

    def get_W(self):
        return self.__W

