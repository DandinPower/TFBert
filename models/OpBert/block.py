from .attention import MultiHeadAttention
from .layer import AddNorm,PositionWiseFFN
import tensorflow as tf
import time
from ..train.timer import GetTimeByDict

class EncoderBlock(tf.keras.Model):
    def __init__(self,config, parameters, logger, index,use_bias=False):
        super(EncoderBlock, self).__init__()
        self.logger = logger
        self.index = index 
        self.config = config
        self.parameters = parameters
        self.attention = MultiHeadAttention(config,parameters,logger,index,use_bias)
        self.addnorm1 = AddNorm(config.dropout, logger)
        self.ffn = PositionWiseFFN(config,parameters, logger,index)
        self.addnorm2 = AddNorm(config.dropout, logger)

    def call(self, inputs):
        (X, valid_lens) = inputs
        output = self.attention((X, X, X, valid_lens))
        Y = self.addnorm1((X, output))
        output = self.ffn(Y)
        result = self.addnorm2((Y, output))
        return result

    def LoadParameters(self):
        self.ffn.LoadParameters()
        self.attention.LoadParameters()