from NeuralNetwork.Layer import Dense, Linear, CrossEntropySoftmax, Sigmoid, Relu
from NeuralNetwork.Model import SingletonSequential
from NeuralNetwork.Loss import CategoricalCrossEntropySoftmax, SSE
from NeuralNetwork.Optimizer import SGD, Adam, AdaMax
import numpy as np
import pickle

import dill
import time

class word2vec:
	def __init__(self, vocabulary, vector_size=50):
		self.vocab = vocabulary
		self.encoder = Dense((vocabulary.count, vector_size), optimizer=AdaMax)
		self.decoder = Dense((vector_size, vocabulary.count), optimizer=AdaMax)
		self.network = SingletonSequential([self.encoder, self.decoder, CrossEntropySoftmax()])
		self.loss = CategoricalCrossEntropySoftmax()
		self.pkl_fit_count = 0

	def train(self, x, y):
		derr, loss = self.network.singleton_train(x, y, self.loss)
		return loss
		"""
		start = time.time()
		z = self.network.batch_forward(x)
		derr = self.loss.batch_dloss(y, z)
		self.network.batch_backward(x, derr)
		stop = time.time()
		print(stop-start)
		return self.loss.batch_loss(y, z)
		"""
		"""
		x = np.array(self.vocab.get_one_hot_dic(x))
		y = np.array(self.vocab.get_one_hot_dic(y))
		z = self.network.forward(x)
		derr = self.loss.dloss(y, z)
		self.network.backward(x, derr)
		return self.loss.loss(y, z)
		"""
	def fit(self, w_inputs, w_targets, max_epoch=1, min_loss=1e-10, batch=32, save_interval=25, verbose=True):
		epoch = 0
		loss = float("inf")
		train_data = [*zip(w_inputs, w_targets)]
		train_data = np.array(train_data[:-(len(train_data) % batch) or None]).reshape(int(len(train_data)/batch), batch, 2).tolist() + [train_data[-(len(train_data) % batch):]]	
		while(epoch < max_epoch and loss > min_loss):
			loss = 0
			"""
			for index, (w_input, w_target) in enumerate(train_data):
				l = self.train(w_input, w_target)
				loss += l
				if verbose:
					print(epoch, str(index+1)+"/"+str(len(train_data)), loss/(index+1), l, end='\r')
				if index % batch == 0 or index +1 == len(train_data):
					self.save("w2v_temp.pkl")
					self.network.epoch_end() 
			epoch += 1
			print()
			"""
			"""
			batchX = []
			batchY = []
			for index, (w_input, w_target) in enumerate(train_data):
				if (index+1) % batch != 0 and (index+1) != len(train_data):
					batchX.append(w_input)
					batchY.append(w_target)
				if (index+1) % batch == 0 or (index+1) == len(train_data):
					x = self.vocab.get_one_hot_dic_batch(batchX)
					y = self.vocab.get_one_hot_dic_batch(batchY)
					l = self.train(x, y)
					loss += l
					print(epoch, str(index+1)+"/"+str(len(train_data)), loss/(index+1), l/batch, end='\r')
					batchX = []
					batchY = []	
					self.network.epoch_end() 
			self.save("w2v_temp.pkl")
			epoch +=1
			print()
			"""
			#"""
			for index, data in enumerate(train_data):
				x, y = zip(*data)
				x = self.vocab.get_one_hot_dic_batch(x)
				y = self.vocab.get_one_hot_dic_batch(y)
				l = self.train(x, y)
				loss += l
				print(epoch, str(index+1)+"/"+str(len(train_data)), loss/((index+1)*batch), l/batch, end='\r')
				if index % save_interval == 0:
					self.save("w2v_temp.pkl")
			epoch +=1
			print()
			#"""

	def pkl_fit(self, file, max_epoch=1, min_loss=1e-1, batch=32, verbose=True):
		epoch = 0
		loss = float("inf")
		while(epoch < max_epoch and loss > min_loss):
			loss = 0
			iteration = True
			index = 0
			for i in range(self.pkl_fit_count):
				for i in range(20000):
					try:
						i, o = pickle.load(file)
					except:
						pass
			while iteration:
				x = []	
				y = []
				for i in range(20000):
					try:
						i, o = pickle.load(file)
						x.append(i)
						y.append(o)
						index += 1
					except Exception as e:
						iteration=False
						break
				self.pkl_fit_count +=1
				self.fit(x, y, max_epoch, min_loss, batch, verbose)
			self.pkl_fit_count=0
			epoch +=1
			print()
	def vectorize(self, w_input):
		return self.encoder.forward(self.vocab.get_one_hot_dic(w_input))

	def batch_vectorize(self, w_input):
		ret = self.encoder.batch_forward(self.vocab.get_one_hot_dic_batch(w_input))
		return ret

	def save(self, filedir):
		dill.dump(self, open(filedir, "wb"))

	def load(filedir):
		return dill.load(open(filedir, "rb"))
