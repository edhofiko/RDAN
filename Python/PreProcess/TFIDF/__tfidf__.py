import numpy as np
import math

class TFIDF:
	def __init__(self, vocabulary=None):
		self.vocabulary = vocabulary
		self.IDF = None
		self.fitted = False
	def __call__(self, corpus):
		if self.vocabulary is None:
			self.vocabulary = Vocabulary()
		if not self.vocabulary.fitted:
			self.vocabulary(corpus)
		self.IDF = IDF.calculate(corpus, self.vocabulary.dictionary)

	def fit(self, corpus):
		self.__call__(corpus)
	
	def transform(self, doc):
		TF = np.zeros(self.vocabulary.count)		
		for sentence in doc:
				words = sentence.split()
				for word in words:
					index = self.vocabulary.dic_get(word)
					if index != -1:
						TF[index] += 1
		return TF * self.IDF
	
	def trnasform_corpus(self, corpus):
		tfidf = []
		for doc in corpus:
			tfidf.append(self.transform(doc))
		return tfidf

class IDF:
	def calculate(corpus, vocabulary=None):
		df = {}
		#Word Count (document frequency)
		for doc in corpus:
			sentences = doc.split("\n")
			w = {}
			for sentence in sentences:
				words = sentence.split()
				for word in words:
					if vocabulary is not None:
						if word in vocabulary:
							w[word] = 1
					else:
						w[word] = 1
			for word, e in w.items():
				df[word] = df.get(word, 0) + 1

		#Word Scoring (inverted document frequency)	
		idf = []		
		for word, count in df.items():
			idf.append([word, math.log(len(corpus)/count)])
		del df
		return idf

	def __call__(self, corpus):
		return IDF.calculate(corpus)

		

