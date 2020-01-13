from Classifier import CNNGRUTextClassifier
from Vectorizer import word2vec
from NeuralNetwork.Layer import OneHotter
import random
import pandas as pd
import numpy as np


vectorizer = word2vec.load("data/word2vec_skipgram.pkl")

data = pd.read_csv("data/data_berita_prep.csv")
gps = data.groupby(["kanal"]).head(10).reset_index(drop=True)

#l = gps["body"].map(lambda x : max([len(x.split("--"))]))

data = gps.values
np.random.shuffle(data)
X, Y = zip(*gps.values)
onehotter = OneHotter(Y)
Y = onehotter(Y)

del data

classifier = CNNGRUTextClassifier(vectorizer, 50, 70, 14)
classifier.fit(X, Y)

