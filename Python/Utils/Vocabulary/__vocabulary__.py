import pickle
import numpy as np
import math
from PreProcess import IDF

class Vocabulary:
	def __init__(self):
		self.dictionary = {}
		self.inverted_dictionary = []
		self.count = 0
		self.fitted = False
		self.idf = {}

	def __call__(self, corpus, max_count=None):
		#IDF
		idf = IDF.calculate(corpus)
		#Make dictionary
		idf = list(sorted(idf, key = lambda x : x[1], reverse=True))						
		max_count = max_count if max_count is not None and max_count < len(idf) else len(idf)
		vocab_idf = {}
		for word, idf in idf:
			if self.count >= max_count:
				break
			self.add_word(word, idf)
		self.add_word("-UNK-", 0)
		self.add_word("-SOS-", 0)
		self.add_word("-EOS-", 0)
		self.add_word("-NUMBER-", 0)
		return self.idf

	def fit(self, corpus, max_count=None):
		return self.__call__(corpus, max_count)	

	def add_word(self, word, idf):
		if word not in self.dictionary:
			self.dictionary[word] = self.count
			self.inverted_dictionary.append(word)
			self.idf[word] = idf
			self.count += 1
			self.fitted = True

	def dic_get(self, word):
		return self.dictionary.get(word, self.dictionary["-UNK-"])
	
	def inv_dic_get(self, index):
		return self.inverted_dictionary[index]
	
	def get_one_hot(self, index):
		one_hot = np.zeros((1, self.count))
		one_hot[0, index] = 1
		return one_hot

	def get_one_hot_dic(self, word):
		return self.get_one_hot(self.dic_get(word))				
	
	def get_one_hot_batch(self, indexes):
		one_hots = np.zeros((len(indexes), self.count))
		for i, index in enumerate(indexes):
			one_hots[i, index] = 1
		return one_hots

	def get_one_hot_dic_batch(self, words):
		indexes = []
		for word in words:
			indexes.append(self.dic_get(word))
		return self.get_one_hot_batch(indexes)

	def load(filedir):
		f = open(filedir, "rb")
		obj = pickle.load(f)
		if not isinstance(obj, Vocabulary):
			raise Exception("Not a Vocabulary")
		return obj

	def save(self, filedir):
		f = open(filedir, "wb")
		pickle.dump(self, f)


