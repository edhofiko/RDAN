from NeuralNetwork.Layer import Dense, Convolve1d, GRU, Tanh, Relu, CrossEntropySoftmax, RNN, Softmax, biGRU
from NeuralNetwork.Loss import CategoricalCrossEntropySoftmax, SSE
from NeuralNetwork.Model import Sequential
from NeuralNetwork.Optimizer import AdaMax
import dill
import numpy as np


class CNNGRUTextClassifier:
	def __init__(self, vectorizer, word_vector_size, max_sentence_length, num_of_class, kernel_size=5, kernel_count=15, hidden_size=20):
		self.vectorizer = vectorizer
		self.convolve_len = max_sentence_length
		self.layer1 = Convolve1d((kernel_size, word_vector_size), kernel_count)
		self.layer2 = LeakyRelu()
		self.layer5 = GRU((max_sentence_length - kernel_size+1, hidden_size), s2v=True)
		self.layer6 = Dense((hidden_size, num_of_class))
		self.layer7 = CrossEntropySoftmax()
		self.model1 = Sequential([self.layer2])
		self.model2 = Sequential([self.layer5, self.layer6, self.layer7])
		self.Loss = CategoricalCrossEntropySoftmax()

	def transform_input(self, doc):
		transformed = [np.array([self.vectorizer.batch_vectorize(sentence.split())[0]])  for sentence in doc.split("--")]
		return self.layer1.batch_forward(np.array([np.pad(sentence, ((0,self.convolve_len - sentence.shape[0]), (0,0)))  for sentence in transformed]))

	def forward(self, x):
		return self.model2.forward(self.model1.batch_forward(x))

	def train(self, x, y):
		z = self.model1.batch_forward(x)
		p = self.model2.forward(z)
		derr = self.Loss.dloss(y, p)
		derr = self.model2.backward(z, derr)
		derr = self.model1.batch_backward(x, derr)
		return self.Loss.loss(y, p), np.argmax(p), np.argmax(y)

	def fit(self, corpus, y, max_epoch=500, min_error=1e-1, verbose=True):
		epoch = 0
		loss = float("inf")	
		corpus = [self.transform_input(doc) for doc in corpus]
		train_data = list(zip(corpus, y))
		while (epoch < max_epoch and loss > min_error):
			loss = 0
			acc = 0
			for index, (x, t) in enumerate(train_data):
				l, p, t =  self.train(x, t)
				loss += l
				acc += 1 if p == t else 0
				print(epoch, str(index+1)+"/"+str(len(corpus)), loss/(index+1), acc/(index+1), end='\r')
			#self.save("CNN-GRU_Cassifier_temp.pkl")
			print()
			epoch+=1
		self.save("data/CNN-GRU_Cassifier.pkl")
			
	def save(self, filedir):
		dill.dump(self, open(filedir, "wb"))

	def load(filedir):
		return dill.load(open(filedir, "rb"))

class HierGRUTextClassifier:
	def __init__(self, vectorizer, word_vector_size, num_of_class, hidden_size=10):
		self.vectorizer = vectorizer
		self.layer1 = GRU((word_vector_size, word_vector_size), s2v=True)
		self.layer5 = GRU((word_vector_size, 2*word_vector_size), s2v=True)
		self.layer6 = Dense((2*word_vector_size, hidden_size))
		self.layer7 = Relu()
		self.layer10 = Dense((hidden_size, num_of_class))
		self.layer11 = CrossEntropySoftmax()
		self.model1 = Sequential([self.layer1])
		self.model2 = Sequential([self.layer5, self.layer6, self.layer7, self.layer10, self.layer11])
		self.Loss = CategoricalCrossEntropySoftmax()

	def transform_input(self, doc):
		transformed = np.array([[self.vectorizer.batch_vectorize(sentence.split())][0]  for sentence in doc.split("--")])
		return transformed
		

	def forward(self, x):
		return self.model2.forward(self.model1.batch_forward(x))

	def train(self, x, y):
		z = self.model1.batch_forward(x)	
		p = self.model2.forward(z)
		derr = self.Loss.dloss(y, p)
		derr = self.model2.backward(z, derr)
		derr = self.model1.batch_backward(x, derr)
		return self.Loss.loss(y, p) / len(p), np.argmax(p), np.argmax(y)

	def fit(self, corpus, y, max_epoch=50000000, min_error=1e-1, verbose=True):
		epoch = 0
		loss = float("inf")	
		corpus = [self.transform_input(doc) for doc in corpus]
		train_data = list(zip(corpus, y))
		while (epoch < max_epoch and loss > min_error):
			loss = 0
			acc = 0
			for index, (x, t) in enumerate(train_data):
				l, p, t =  self.train(x, t)
				loss += l
				acc += 1 if p == t else 0
				print(epoch, str(index+1)+"/"+str(len(corpus)), loss/(index+1), t,p,acc/(index+1), end='\r')
			#self.save("CNN-GRU_Cassifier_temp.pkl")
			print()
			epoch+=1
		self.save("data/CNN-GRU_Cassifier.pkl")
			
	def save(self, filedir):
		dill.dump(self, open(filedir, "wb"))

	def load(filedir):
		return dill.load(open(filedir, "rb"))
		
		
