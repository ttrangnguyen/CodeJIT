import re
import sys
import os

import warnings
warnings.filterwarnings("ignore")
from gensim.models import Word2Vec

class Word2VecTrain:
    def __init__(self, vocabularies, vector_length, embedding_file):
        self.gadgets = vocabularies
        self.vector_length = vector_length
        self.vector_file = embedding_file
        print("----------")
        print("Word2Vec")
        print("vector_length:", vector_length)
        print("----------")

    def train_model(self):
        model = Word2Vec(self.gadgets, min_count=1, size=self.vector_length, sg=1)
        print("Vocab size:", len(model.wv.vocab))
        self.embeddings = model.wv
        self.embeddings.save(self.vector_file)
        del model
        del self.gadgets