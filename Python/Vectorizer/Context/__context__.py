class AbstractContext:
	def __init__(self, vocab):
		self.vocab = vocab

	def _abstract_op(self, context, word):
		pass	
	
	def transform(self, sentence,  context_count=1):
		tokens = sentence.split()
		dic = self.vocab.dictionary
		for i_w in range(len(tokens)):
			for i_b in range(1, context_count+1):
				i_c = i_w - i_b
				if i_c > 0:
					if tokens[i_c] in dic and tokens[i_w] in dic:
						yield self._abstract_op(tokens[i_c], tokens[i_w])
			for i_a in range(1, context_count+1):
				i_c = i_w + i_b
				if i_c < len(tokens):
					if tokens[i_c] in dic and tokens[i_w] in dic:
						yield self._abstract_op(tokens[i_c], tokens[i_w])

	def doc_transform(self, doc, context_count=1, unique=True):
		sentences = doc.split("--")
		t = []
		for sentence in sentences:
			t.extend(list(self.transform(sentence, context_count=context_count)))
		if unique:
			return list(set(t))
		else:
			return t

	def corpus_transform(self, corpus, context_count=1, unique=True):
		t = []
		for doc in corpus:
			t.extend(self.doc_transform(doc, context_count, unique))
		if unique:
			return list(set(t))
		else:
			return t

	def to_vector(self, t1, t2):
		return (self.vocab.get_one_hot_dic(t1), self.vocab.get_one_hot_dic(t2))        
		

class CBOW(AbstractContext):
	def _abstract_op(self, context, word):
		return (context, word)

class SkipGram(AbstractContext):
	def _abstract_op(self, context, word):
		return (word, context)
