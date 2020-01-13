import numpy as np

def convolve1d(self, X, kernel, operation=np.multiply, agregate=np.sum):
	m, _ = kernel.shape
        y, _ = C.shape
        offset = math.top(m/2) if offset is None else offset
        Z = np.empty(size=(y-offset))
        for i in range(0, y-offset):
		Z[i] = agregate(operation(X[i:i+m,:], kernel)
        return new_img   

def dconvolve1d_dX(self, W, X, kernel):
	pass

def dconvolce1d_dW(self, W, X, kernel):
	pass
