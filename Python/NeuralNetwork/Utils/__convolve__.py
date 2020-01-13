import numpy as np
import math

def convolve1d(X, kernel):
	k, _ = kernel.shape
	w, _ = X.shape
	o = w - k + 1
	Z = np.empty((o))
	for i in range(0, o):
		Z[i] = (X[i:i+k or None] * kernel).sum()
	return Z   


def dconvolve1d(X, kernel, derr):
	k, _ = kernel.shape
	w, _ = X.shape
	o = w - k + 1
	dx, dw = np.zeros_like(X), np.zeros_like(kernel)
	for i in range(0, o):
		dx[i:i+k, :] += kernel * derr[i]
		dw += X[i:i+k] * derr[i]
	return dw, dx
