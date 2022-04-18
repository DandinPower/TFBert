from .attention import MultiHeadAttention
from .layer import AddNorm,PositionWiseFFN
import tensorflow as tf
import time
from ..train.timer import GetTimeByDict

class EncoderBlock(tf.keras.Model):
    def __init__(self,config, parameters, index,use_bias=False):
        super(EncoderBlock, self).__init__()
        self.index = index 
        self.config = config
        self.parameters = parameters
        self.attention = MultiHeadAttention(config,parameters,index,use_bias)
        self.addnorm1 = AddNorm(config.dropout)
        self.ffn = PositionWiseFFN(config,parameters,index)
        self.addnorm2 = AddNorm(config.dropout)

    def call(self, inputs):
        (X, valid_lens) = inputs
        start = time.time()
        output = self.attention((X, X, X, valid_lens))
        attentionTime = time.time()
        print(f'attention: {attentionTime-start}')
        Y = self.addnorm1((X, output))
        addnorm1Time = time.time()
        print(f'addnorm1: {addnorm1Time - attentionTime}')
        output = self.ffn(Y)
        ffnTime = time.time()
        print(f'ffn: {ffnTime-addnorm1Time}')
        result = self.addnorm2((Y, output))
        addnorm2Time = time.time()
        print(f'addnorm2: {addnorm2Time - ffnTime}')
        return result

    def LoadParameters(self):
        self.ffn.LoadParameters()
        self.attention.LoadParameters()