import pandas as pd
from Vectorizer import word2vec
from Utils import Vocabulary

data = pd.read_csv("data/data_skipgram.csv")
w_inputs = data["input"].values[:]
w_targets = data["target"].values[:]

vocab = Vocabulary.load("data/vocabulary.pkl")
vectorizer = word2vec.load("data/word2vec_skipgram.pkl")#word2vec(vocab)
vectorizer.fit(w_inputs, w_targets, verbose=True, batch=128)
vectorizer.save("data/word2vec_skipgram.pkl")

