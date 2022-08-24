from .encoder import BERTEncoder
from .block import EncoderBlock
from .layer import LinearLayer
import tensorflow as tf
import time 
from ..train.timer import GetTimeByDict

class BERTModel(tf.keras.Model):
    def __init__(self, config, parameters, logger):
        super(BERTModel, self).__init__()
        self.logger = logger
        self.parameters = parameters
        self.encoder = BERTEncoder(config,parameters, logger)
        self.block1 = EncoderBlock(config,parameters, logger, 0,True)
        self.block2 = EncoderBlock(config,parameters, logger, 1,True)
        self.hidden = tf.keras.Sequential()
        tempLinearLayer = LinearLayer(config.numHiddens, config.numHiddens)
        tempLinearLayer.set_weights([parameters["hidden.0.weight"],parameters["hidden.0.bias"]])
        self.hidden.add(tempLinearLayer)
        self.hidden.add(tf.keras.layers.Activation('tanh'))
        self.numHiddens = config.numHiddens

    def call(self, inputs):
        (tokens, segments, valid_lens) = inputs
        embeddingX = self.encoder((tokens,segments))
        X = self.block1((embeddingX, valid_lens))
        X = self.block2((X, valid_lens))
        self.logger.AddNewLog([X[:, 0, :].shape, [self.numHiddens, self.numHiddens]], "matmul")
        X = self.hidden(X[:, 0, :])
        return X

    def LoadParameters(self):
        self.encoder.LoadParameters()
        self.block1.LoadParameters()
        self.block2.LoadParameters()

class OPBERTClassifier(tf.keras.Model):
    def __init__(self, config, parameters, logger):
        super(OPBERTClassifier, self).__init__()
        self.config = config 
        self.parameters = parameters
        self.logger = logger
        self.bert = BERTModel(config, self.parameters, logger)
        self.classifier = LinearLayer(config.numHiddens, 2)
        self.numHiddens = config.numHiddens

    def call(self, tokens):
        tempSegments = tokens * 0
        tempValid = self.GetValidLen(tokens)
        inputs = (tokens,tempSegments,tempValid)
        output = self.bert(inputs)
        self.logger.AddNewLog([output.shape, [self.numHiddens, 2]], "matmul")
        output = self.classifier(output)
        result = tf.nn.softmax(output)
        return result

    def GetValidLen(self,inputs):
        tokens = inputs
        padding = tf.constant(1)
        temp = tf.math.equal(tokens, padding)
        paddingNums = tf.math.count_nonzero(temp,axis=1,dtype=tf.dtypes.float32)
        paddingNums = self.config.maxLen - paddingNums
        return paddingNums

    def LoadParameters(self):
        self.bert.LoadParameters()


    


    