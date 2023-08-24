import re
import sys
import os

import warnings
warnings.filterwarnings("ignore")

from gensim.models import Word2Vec
from gensim.models import FastText
import numpy
from vul_localization.helper import *
CTX_SURROUNDING = "ctx_surrounding"
CTX_CDG = "ctx_cfg"
CTX_DDG = "ctx_ddg"
CTX_OPERATION = "ctx_operation"
VUL_TYPE = "vul_type"

# Sets for operators


vulnerability_types = ['DoS', 'Exec Code', '+Priv',
                      'Overflow', 'Mem. Corr.', 'Bypass', 'Dir. Trav.', 
                      '+Info', 'Http', 'R.Spl.', 'XSS', 'Other']

"""
Functionality to train Word2Vec model and vectorize gadgets
Buffers list of tokenized gadgets in memory
Trains Word2Vec model using list of tokenized gadgets
Uses trained model embeddings to create 2D gadget vectors
"""
class GadgetVectorizer:

    def __init__(self, wv, vector_length, max_seq_length, max_code_stmt_length):
        self.gadgets = []
        self.vector_length = vector_length
        self.max_seq_length = max_seq_length
        self.max_code_stmt_length = max_code_stmt_length
        self.embeddings = wv
        print("----------")
        print("GadgetVectorizer")
        print("vector_length:", vector_length)
        print("max_seq_length:", max_seq_length)
        print("max_code_stmt_length:", max_code_stmt_length)
        print("----------")

        
    def vectorize_vul_type(vul_type):
        vectors = numpy.zeros(shape=(1, 12))
        if not isinstance(vul_type,str):
          vectors[0][-1] = 1
          return vectors
        for i in range(0, len(vulnerability_types)):
          t = vulnerability_types[i]
          if t in vul_type:
            vectors[0][i] = 1
        return vectors

    """
    Uses Word2Vec to create a vector for each gadget
    Gets a vector for the gadget by combining token embeddings
    Number of tokens used is min of number_of_tokens and 50
    """
       
    def vectorize(self, cdg_context, ddg_context, operation_ctx_abstract): 
        cdg_context_vectors = numpy.zeros(shape=(self.max_seq_length, self.vector_length))
        tokenized = tokenize(cdg_context)
        for i in range(min(len(tokenized), self.max_seq_length)):
          if tokenized[i] in self.embeddings:
            cdg_context_vectors[i] = self.embeddings[tokenized[i]]
          
        ddg_context_vectors = numpy.zeros(shape=(self.max_seq_length, self.vector_length))
        tokenized = tokenize(ddg_context)
        for i in range(min(len(tokenized), self.max_seq_length)):
          if tokenized[i] in self.embeddings:
            ddg_context_vectors[i] = self.embeddings[tokenized[i]]

        operation_ctx_abstract_vectors = numpy.zeros(shape=(self.max_code_stmt_length, self.vector_length))
        tokenized = tokenize(operation_ctx_abstract)
        for i in range(min(len(tokenized), self.max_code_stmt_length)):
          if tokenized[i] in self.embeddings:
            operation_ctx_abstract_vectors[i] = self.embeddings[tokenized[i]]
        
        return [cdg_context_vectors, ddg_context_vectors, operation_ctx_abstract_vectors]
       
       
           
   