import tensorflow as tf
class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim,output_dim):
        super().__init__()
        self.w = self.add_variable(name='w',
            shape=[input_dim, output_dim], initializer=tf.zeros_initializer())
        self.b = self.add_variable(name='b',
            shape=[output_dim], initializer=tf.zeros_initializer())

    def call(self, inputs):
        y_pred = tf.matmul(inputs, self.w) + self.b
        return y_pred

class AddNorm(tf.keras.Model):
    def __init__(self, dropout, logger):
        super(AddNorm, self).__init__()
        self.logger = logger
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.ln = tf.keras.layers.LayerNormalization(axis = 2)

    def call(self, inputs):
        (X,Y) = inputs
        return self.ln(self.dropout(Y) + X)

class PositionWiseFFN(tf.keras.Model):
    def __init__(self, config, parameters, logger,index):
        super(PositionWiseFFN, self).__init__()
        self.logger = logger
        self.config = config 
        self.parameters = parameters 
        self.index = index 
        self.dense1 = LinearLayer(config.ffnNumInput, config.ffnNumHiddens)
        self.relu = tf.keras.layers.ReLU()
        self.dense2 = LinearLayer(config.ffnNumHiddens, config.ffnNumInput)

    def call(self, X):
        self.logger.AddNewLog([X.shape, self.dense1.w.shape], "matmul")
        output = self.relu(self.dense1(X))
        self.logger.AddNewLog([output.shape, self.dense2.w.shape], "matmul")
        return self.dense2(output)

    def LoadParameters(self):
        self.dense2.set_weights([self.parameters[f"encoder.blks.{self.index}.ffn.dense1.weight"],self.parameters[f"encoder.blks.{self.index}.ffn.dense2.bias"]])
        self.dense1.set_weights([self.parameters[f"encoder.blks.{self.index}.ffn.dense2.weight"],self.parameters[f"encoder.blks.{self.index}.ffn.dense1.bias"]])