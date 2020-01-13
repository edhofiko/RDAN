from PreProcess import DataPrep
from Utils import Vocabulary
from Vectorizer import CBOW, SkipGram
import pandas as pd

#Process Data
data = pd.read_csv("data/data_berita_1.csv", engine="python").head(10000)
data["body"] = DataPrep.prep(data["body"])
data = data[["body", "kanal"]]
data.to_csv("data/data_berita_prep.csv", index=False)

#Vocabulary
corpus = data["body"].values
vocab = Vocabulary()
vocab.fit(corpus, max_count=500000)
vocab.save("data/vocabulary.pkl")

"""
#CBOW
cbow = CBOW(vocab)
data_cbow = cbow.corpus_transform(corpus, context_count=2)
data_cbow_pd = pd.DataFrame(data_cbow, columns=["input", "target"])
data_cbow_pd.to_csv("data/data_cbow.csv", index=False)
"""

#Skip-Gram
skipgram = SkipGram(vocab)
data_skipgram = skipgram.corpus_transform(corpus, context_count=3)
data_skipgram_pd = pd.DataFrame(data_skipgram, columns=["input", "target"])
data_skipgram_pd.to_csv("data/data_skipgram.csv", index=False)

