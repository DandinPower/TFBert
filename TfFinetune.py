from models.bert.configs import Config
from models.bert.modeling import BERTModel,BERTClassifier
from models.preprocess.data import YelpDataset,load_vocab,DataLoader,GetTrainDataset,GetTestDataset
from models.preprocess.load import load_variable,Parameters,LoadModel,SaveModel,WriteTfLite,WriteInt8TFLite
from models.train.classification import Train,Inference
from models.train.multigpu import MultiTrain
from models.valid.tflite import TfliteTest
from dotenv import load_dotenv
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
load_dotenv()

PARAMETER_PATH = os.getenv('PARAMETER_PATH')
MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH')
DATASET_PATH = os.getenv('DATASET_PATH')
MAX_LEN = int(os.getenv('MAX_LEN'))
SPLIT_RATE = float(os.getenv('SPLIT_RATE'))
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS'))
LR = float(os.getenv('LR'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
GPU_NUMS = int(os.getenv('GPU_NUMS'))
GLOBAL_BATCH_SIZE = GPU_NUMS * BATCH_SIZE
TFLITE_PATH = os.getenv('TFLITE_PATH')
TFLITE_INT8_PATH = os.getenv('TFLITE_INT8_PATH')

def DataFlowTest():
    config = Config()
    parameters = load_variable(PARAMETER_PATH)
    parameters = Parameters(parameters)
    model = BERTClassifier(config, parameters)
    model.LoadParameters()
    datas,labels = GetSingleDataset(DATASET_PATH,MAX_LEN,SPLIT_RATE)
    singleData = datas[1]
    singleLabels = labels[1]
    print(f'Input Token: ')
    print(singleData)
    output = model(singleData)
    print(output)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=singleLabels, y_pred=output)
    loss = tf.reduce_mean(loss) 
    print(loss)

def SingleTest():
    config = Config()
    datas,labels = GetSingleDataset(DATASET_PATH,MAX_LEN,SPLIT_RATE)
    singleData = datas[1]
    singleLabels = labels[1]
    print(f'Input Token: ')
    print(singleData)
    newModel = LoadModel(MODEL_SAVE_PATH)
    output = newModel(singleData)
    print(output)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=singleLabels, y_pred=output)
    loss = tf.reduce_mean(loss) 
    print(loss)

def OnlyInference():
    dataset = GetTestDataset(DATASET_PATH,MAX_LEN,SPLIT_RATE,BATCH_SIZE)
    newModel = LoadModel(MODEL_SAVE_PATH)
    Inference(newModel,dataset)

def MultiTest():
    config = Config()
    parameters = load_variable(PARAMETER_PATH)
    parameters = Parameters(parameters)
    print("Load Multi Dataset...")
    dataset_multi = GetTrainDataset(DATASET_PATH,MAX_LEN,SPLIT_RATE,GLOBAL_BATCH_SIZE)
    print("Load Normal Dataset...")
    dataset_one = GetTrainDataset(DATASET_PATH,MAX_LEN,SPLIT_RATE,BATCH_SIZE)
    devices = ["/gpu:0","/gpu:1"]
    model = MultiTrain(config, parameters, devices, dataset_multi, LR, NUM_EPOCHS,MODEL_SAVE_PATH)
    Inference(model,dataset_one)
    devices = ["/gpu:0"]
    model = MultiTrain(config, parameters, devices, dataset_one, LR, NUM_EPOCHS,MODEL_SAVE_PATH)
    Inference(model,dataset_one)


def main():
    config = Config()
    parameters = load_variable(PARAMETER_PATH)
    parameters = Parameters(parameters)
    model = BERTClassifier(config, parameters)
    model.LoadParameters()
    dataset = GetTrainDataset(DATASET_PATH,MAX_LEN,SPLIT_RATE,BATCH_SIZE)
    model = Train(model,dataset, LR, NUM_EPOCHS,MODEL_SAVE_PATH)
    testDataset = GetTestDataset(DATASET_PATH,MAX_LEN,SPLIT_RATE,BATCH_SIZE)
    Inference(model,testDataset)
    SaveModel(model, MODEL_SAVE_PATH)
    newModel = LoadModel(MODEL_SAVE_PATH)
    Inference(newModel,testDataset)

if __name__ == "__main__":
    MultiTest()
    #DataFlowTest()
    #SingleTest()
    main()
    OnlyInference()
    #WriteTfLite(MODEL_SAVE_PATH, TFLITE_PATH)
    #WriteInt8TFLite(MODEL_SAVE_PATH, TFLITE_INT8_PATH)
    #TfliteTest()