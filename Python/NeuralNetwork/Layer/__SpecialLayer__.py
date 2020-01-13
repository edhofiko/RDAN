from NeuralNetwork.Layer import AbstractLayer
import numpy as np

class OneHotter:
	def __init__(self, Y):
		unique = set(Y)
		dic = {}

		for i, d in enumerate(unique):
			dic[d] = i
		counter = [0] * len(dic.keys())	
		for y in Y:
			counter[dic[y]] += 1
		print(unique)
		print(counter)
		self.dic = dic

	def __call__(self, Y):
		ret = np.zeros((len(Y), len(self.dic.keys())))
		for i, d in enumerate(Y):
			ret[i, self.dic[d]] = 1

		return ret
