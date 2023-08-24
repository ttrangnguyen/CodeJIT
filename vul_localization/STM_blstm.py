from __future__ import print_function

import warnings
from pandas.core.groupby.grouper import Grouper

from tensorflow.keras.utils import to_categorical
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, Bidirectional, LeakyReLU, GRU, GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate, Input, Conv1D, MaxPooling1D, Flatten, Maximum, Minimum, Add, Average, TimeDistributed, Flatten, Masking
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pickle
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from STM_classification_manager import * 
import tensorflow as tf
# from attention import Attention

def generate_data(vectorizer, data_file, batch_size, epochs): 
    data = pd.read_csv(data_file, encoding='utf-8') 
    i = 0
    ep = 0
    while ep < epochs:
      i = 0
      while i < data.shape[0]:
          batch = []
          for b in range(batch_size):
            if i < data.shape[0]:
              sample_vector =  get_vector_a_sample(vectorizer, data.iloc[i])
              batch.append(sample_vector)
              i += 1
          df = pd.DataFrame(batch)
          X = np.stack(df.iloc[:, 0].values)
          y = df.iloc[:, 1].values 
          # if len(np.unique(y)) > 1:
          #   oversample = RandomOverSampler(sampling_strategy='minority')
          #   #oversample = SMOTE()
          #   X_over, y_over = oversample.fit_resample(X, y)
          #   X = X_over
          #   y = y_over
          X = unzip_X_inputs(X)
          yield X, to_categorical(y)
      ep += 1

def unzip_X_inputs(X):
    
    X_training_df = pd.DataFrame(X, columns = ["cdg_ctx_training",  "ddg_ctx_training",  "operation_ctx_training"])
    cdg_ctx_X_train = np.stack(X_training_df.loc[:, 'cdg_ctx_training'].to_numpy())
    ddg_ctx_X_train = np.stack(X_training_df.loc[:, 'ddg_ctx_training'].to_numpy())
    operation_ctx_X_train = np.stack(X_training_df.loc[:, 'operation_ctx_training'].to_numpy())
   
    return [cdg_ctx_X_train, ddg_ctx_X_train, operation_ctx_X_train ]
    
    

  
"""
Bidirectional LSTM neural network
Structure consists of two hidden layers and a BLSTM layer
Parameters, as from the VulDeePecker paper:
    Nodes: 300
    Dropout: 0.5
    Optimizer: Adamax
    Batch size: 64
    Epochs: 4
"""
class STM_Train_BLSTM:
    def __init__(self, vectorizer, data_file, name="", vector_length = 64, max_seq_length = 50, max_code_stmt_length = 40, batch_size=64):
        self.max_seq_length = max_seq_length
        self.max_code_stmt_length = max_code_stmt_length
        self.vector_length = vector_length
        self.name = name
        self.batch_size = batch_size
        self.data_file = data_file
        self.vectorizer = GadgetVectorizer(vectorizer, vector_length, max_seq_length, max_code_stmt_length)
        data = pd.read_csv(data_file, encoding='utf-8')
        self.num_training_samples = data.shape[0] - 1
        data = pd.read_csv(data_file, encoding='utf-8', nrows = 2)
        df = get_vector_data(self.vectorizer, data)
        del data
        X_train = np.stack(df.iloc[:, 0].values)
        print("BiLSTM shape", X_train.shape)
        y_train = df.iloc[:, 1].values
        
        X_train = unzip_X_inputs(X_train)
        print(type(X_train))
        
        #self.class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_train)
        
        print("-----")
        self.X_train = X_train
        self.y_train = to_categorical(y_train)

        # Lower learning rate to prevent divergence
        self.model = self.init_combined_model()

    def init_combined_model(self):
    
        
       
        bw_cdg_ctx_input_shape = self.X_train[0][0].shape
        bw_ddg_ctx_input_shape = self.X_train[1][0].shape
        operation_ctx_input_shape = self.X_train[2][0].shape
       

        lsm_unit = 256
        
        #print("Using Combined LSTM - Input Shape [Surrounding_CTX {}] [CDG_CTX {}] [DDG_CTX {}][Operation CTX {}] [Code STMT {}]".format(surrounding_ctx_input_shape, cdg_ctx_input_shape, ddg_ctx_input_shape, operation_ctx_input_shape, code_stmt_input_shape))
        
       
        model_input_3 = Input(shape=bw_cdg_ctx_input_shape)
        model_3 = Bidirectional(LSTM(lsm_unit, return_sequences=True), input_shape=bw_cdg_ctx_input_shape)(model_input_3)
        model_3 = GlobalMaxPooling1D()(model_3)
        

        model_input_5 = Input(shape=bw_ddg_ctx_input_shape)
        model_5 = Bidirectional(LSTM(lsm_unit, return_sequences=True), input_shape=bw_ddg_ctx_input_shape)(model_input_5)
        model_5 = GlobalMaxPooling1D()(model_5)

        model_input_7 = Input(shape=operation_ctx_input_shape)
        model_7 = Bidirectional(LSTM(lsm_unit//2, return_sequences=True), input_shape=operation_ctx_input_shape)(model_input_7)
        model_7 = GlobalMaxPooling1D()(model_7)

       

      
        out = Concatenate()([ model_3, model_5, model_7]) 
        out = Dense(384)(out)
        out = LeakyReLU()(out)
        out = Dropout(0.1)(out)
        out = Dense(192)(out) 
        out = LeakyReLU()(out)
        out = Dropout(0.1)(out)
        out = Dense(2, activation='softmax')(out)
        
        merged_model = Model(inputs=[model_input_3, model_input_5, model_input_7],outputs=out) 
        adamax = Adamax(learning_rate=0.002)
  
        merged_model.compile(adamax, 'categorical_crossentropy', metrics = ['accuracy']) 
   
        return merged_model

    def load_model(self):
        self.model.load_weights("models/" + self.name + "_model.h5")
        return 

    """
    Trains model based on training data
    # """
    def train(self, epochs=200):
        tf.random.set_seed(12345)
        mcp_save = ModelCheckpoint("models/best_model." +self.name + "_model.hdf5", save_best_only=True, monitor = "accuracy", mode='max', verbose=2)
        reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, min_delta=1e-4, mode='min', verbose=1)
        num_batches =  self.num_training_samples//self.batch_size
        self.model.fit_generator(generator=generate_data(self.vectorizer, self.data_file, self.batch_size, epochs), 
        epochs=epochs,
        steps_per_epoch=num_batches,
        callbacks=[mcp_save, reduce_lr_loss]
        )

class STM_Test_BLSTM:
    def __init__(self, vectorizer, data_file, name="", vector_length = 64, max_seq_length = 50, max_code_stmt_length = 40, batch_size=64):
        self.max_seq_length = max_seq_length
        self.max_code_stmt_length = max_code_stmt_length
        self.vector_length = vector_length
        self.name = name
        self.batch_size = batch_size
        self.vectorizer = GadgetVectorizer(vectorizer, vector_length, max_seq_length, max_code_stmt_length)
        self.data_file = data_file
        data = pd.read_csv(data_file, encoding='utf-8')
        self.num_testing_samples = data.shape[0] - 1
        del data
        
        print("-----")

        print("Loading best model for testing")
        self.model = load_model("models/best_model." +self.name + "_model.hdf5")
        print("model loaded")

    """
    Tests accuracy of model based on test data
    Loads weights from file if no weights are attached to model object
    """
    def test(self):
        predictions = []
        targets = []
        positive_samples = 0
        negative_samples = 0
        sk =  0
        while(sk < self.num_testing_samples):
          print("sk:....", sk)
          data = pd.read_csv(self.data_file, encoding='utf-8', skiprows = sk, nrows = 20*self.batch_size)
          
          #data = pd.read_csv(self.data_file, encoding='utf-8', skiprows = sk, nrows = 100)
          data.columns = ['Unnamed: 0.2', 'Unnamed: 0', 'Unnamed: 0.1', 'commit_id',
                          'line_number', 'operation_ctx', 'cdg_bw_slicing', 'cdg_fw_slicing',
                          'ddg_bw_slicing', 'ddg_fw_slicing', 'vul_stmt']

          df = get_vector_data(self.vectorizer, data)

          X_test = np.stack(df.iloc[:, 0].values)
          X_test = unzip_X_inputs(X_test)
          y_test = df.iloc[:, 1].values 
          positive_samples += len(np.where(y_test==1)[0])
          negative_samples += len(np.where(y_test==0)[0])
          targets.append(y_test)
          #values = self.model.evaluate(X_test, y_test, batch_size=self.batch_size)
          tmp_predict = (self.model.predict(X_test, batch_size=self.batch_size))
          predictions.append(tmp_predict)
          sk += 20*self.batch_size
          print("+++++++++++++++")
          print("sk new: ", sk)
          #break
          
        print("")
        print("Positive test sample:",positive_samples)
        print("Negative test sample:",negative_samples)
        print("")
        return predictions, targets

