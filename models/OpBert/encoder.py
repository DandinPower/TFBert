import tensorflow as tf
import sys
import time 

class AddParameter(tf.keras.layers.Layer):
    def __init__(self, nums,hiddens):
        super().__init__()
        self.w = self.add_variable(name='weight',shape=[nums,hiddens], initializer=tf.zeros_initializer())

    def call(self, inputs):
        return inputs + self.w

class BERTEncoder(tf.keras.Model):
    def __init__(self, config, parameters, logger):
        super(BERTEncoder, self).__init__()
        self.logger = logger
        self.token_embedding = tf.keras.layers.Embedding(config.vocabSize, config.numHiddens,input_length=config.maxLen,weights=[parameters["encoder.token_embedding.weight"]])
        self.segment_embedding = tf.keras.layers.Embedding(2, config.numHiddens,input_length=config.maxLen,weights=[parameters["encoder.segment_embedding.weight"]])
        self.pos_embedding = AddParameter(config.maxLen,config.numHiddens)
        self.config = config
        self.parameters = parameters
        self.maxLen = config.maxLen
        self.vocabSize = config.vocabSize
        self.numHiddens = config.numHiddens

    def call(self, inputs):
        (tokens,segments) = inputs
        X = self.token_embedding(tokens)
        X = X + self.segment_embedding(segments)
        X = self.pos_embedding(X)
        shape1_A = [tokens.shape[0], self.maxLen, self.vocabSize]
        shape1_B = [self.vocabSize, self.numHiddens]
        shape2_A = [segments.shape[0], self.maxLen, 2]
        shape2_B = [2, self.numHiddens]
        self.logger.AddNewLog([shape1_A, shape1_B], "matmul")
        self.logger.AddNewLog([shape2_A, shape2_B], "matmul")
        return X

    def LoadParameters(self):
        self.pos_embedding.set_weights(self.parameters["encoder.pos_embedding"])
        
