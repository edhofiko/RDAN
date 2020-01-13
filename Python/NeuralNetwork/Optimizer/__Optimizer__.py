import numpy as np

class AbstractOptimizer:
    def __init__(self, size, alpha=0.002):
        self._size = size
        self._alpha = alpha
        self._iter = 0
    def __call__(self, dtheta):
        pass
    def update(self):
        pass


class SGD(AbstractOptimizer):
    def __call__(self, dtheta):
        dtheta *= self._alpha
        return dtheta

class Momentum(AbstractOptimizer):
    def __init__(self, size, alpha=0.002, gamma=0.02):
        super().__init__(size, alpha)
        self._gamma = gamma
        self.v = np.zeros(size)
        self._temp = np.zeros(size)

    def __call__(self, dtheta):
        dtheta += self._gamma * self.v
        self.v = dtheta
        return self._alpha * dtheta
  

class AdaMax(AbstractOptimizer):
    def __init__(self, size, alpha=0.002, beta1=0.9, beta2=0.999, epsilon=1e-9):
        super().__init__(size, alpha)
        self._beta1 = beta1
        self._beta2 = beta2
        self._beta1_t = beta1
        self._m_dtheta = np.zeros(size)+epsilon
        self._u_dtheta = np.zeros(size)+epsilon
        self._d_m = np.zeros(size)
        self._d_u = np.zeros(size)
        self._epsilon = epsilon
        self._dtime = 0

    def __call__(self, dtheta):      
        self._m_dtheta = self._beta1 * self._m_dtheta + (1-self._beta1) * dtheta
        self._u_dtheta = np.maximum(self._beta2 * self._u_dtheta, np.abs(dtheta))
        dtheta = (self._alpha/(1-self._beta1_t)) *  self._m_dtheta / ( self._u_dtheta)
        self._beta1_t *= self._beta1     
        return dtheta

class Adam(AbstractOptimizer):
    def __init__(self, size, alpha=0.002, beta1=0.9, beta2=0.999, epsilon=1e-9):
        super().__init__(size, alpha)
        self._beta1 = beta1
        self._beta2 = beta2
        self._beta1_t = beta1
        self._beta2_t = beta2
        self._m_dtheta = np.zeros(size)+epsilon
        self._u_dtheta = np.zeros(size)+epsilon
        self._d_m = np.zeros(size)
        self._d_u = np.zeros(size)
        self._epsilon = epsilon
        self._dtime = 0

    def __call__(self, dtheta):      
        self._m_dtheta = self._beta1 * self._m_dtheta + (1-self._beta1) * dtheta
        self._u_dtheta = self._beta2 * self._u_dtheta + (1-self._beta2) * dtheta ** 2
        m_dtheta = self._m_dtheta / (1 - self._beta1_t)
        u_dtheta = self._u_dtheta / (1 - self._beta2_t)
        dtheta =  self._alpha * m_dtheta / (np.sqrt(u_dtheta) + self._epsilon)
        self._beta1_t *= self._beta1     
        self._beta2_t *= self._beta2
        return dtheta


