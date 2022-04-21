from progressbar import ProgressBar
from ..bert.modeling import BERTClassifier
from ..bert.configs import Config
from ..preprocess.load import load_variable,Parameters
from dotenv import load_dotenv
#import joblib
import os
load_dotenv()
import tensorflow as tf
import time

PARAMETER_PATH = os.getenv('PARAMETER_PATH')
GPU_NUMS = int(os.getenv('GPU_NUMS'))
SINGLE_BATCH = int(os.getenv('SINGLE_BATCH'))
GLOBAL_BATCH_SIZE = GPU_NUMS * SINGLE_BATCH

def compute_loss(labels, predictions):
    per_example_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=labels, y_pred=predictions)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

def MultiTrain(config,parameters,xTrain,yLabel,lr,num_epochs,savePath):
    print('Multi Training...')
    mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
    with mirrored_strategy.scope():
        model = BERTClassifier(config, parameters)
        model.LoadParameters()
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr)  
    
        dataset = tf.data.Dataset.from_tensor_slices((xTrain, yLabel)).batch(GLOBAL_BATCH_SIZE)
        dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

        def train_step(inputs):
            features, labels = inputs
            with tf.GradientTape() as tape:
                predictions = model(features, training=True)
                loss = compute_loss(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        @tf.function
        def distributed_train_step(dist_inputs):
            per_replica_losses = mirrored_strategy.run(train_step, args=(dist_inputs,))
            return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                    axis=None)

        for epoch in range(num_epochs):
            lossTotal = 0
            start = 0
            total = len(dataset)
            for dist_inputs in dist_dataset:
                tempLoss = distributed_train_step(dist_inputs)
                lossTotal += tempLoss
            print(f'epoch: {epoch} loss: {lossTotal/total}')
    print("finish")
