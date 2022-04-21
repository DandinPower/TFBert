from models.bert.configs import Config
from models.bert.modeling import BERTModel,BERTClassifier
from models.preprocess.data import YelpDataset,load_vocab,DataLoader,GetTrainDataset,GetTestDataset,GetSingleDataset,GetNoBatchDataset
from models.preprocess.load import load_variable,Parameters,LoadModel,SaveModel,WriteTfLite,WriteInt8TFLite
from models.train.classification import Train,Inference,InferenceByDataset
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
SINGLE_BATCH = int(os.getenv('SINGLE_BATCH'))

GLOBAL_BATCH_SIZE = GPU_NUMS * SINGLE_BATCH

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
    datas,labels = GetTestDataset(DATASET_PATH,MAX_LEN,SPLIT_RATE,BATCH_SIZE)
    newModel = LoadModel(MODEL_SAVE_PATH)
    Inference(newModel,datas, labels)

def MultiTest():
    config = Config()
    parameters = load_variable(PARAMETER_PATH)
    parameters = Parameters(parameters)
    print("Load Dataset...")
    datas,labels = GetNoBatchDataset(DATASET_PATH,MAX_LEN,SPLIT_RATE,BATCH_SIZE)
    print("Load Multi Dataset...")
    dataset_multi = tf.data.Dataset.from_tensor_slices((datas, labels)).batch(GLOBAL_BATCH_SIZE)
    print("Load Normal Dataset...")
    dataset_one = tf.data.Dataset.from_tensor_slices((datas, labels)).batch(BATCH_SIZE)
    devices = ["/gpu:0","/gpu:1"]
    model = MultiTrain(config, parameters, devices, dataset_multi, LR, NUM_EPOCHS,MODEL_SAVE_PATH)
    InferenceByDataset(model,dataset_one)
    devices = ["/gpu:0"]
    model = MultiTrain(config, parameters, devices, dataset_one, LR, NUM_EPOCHS,MODEL_SAVE_PATH)
    InferenceByDataset(model,dataset_one)


def main():
    config = Config()
    parameters = load_variable(PARAMETER_PATH)
    parameters = Parameters(parameters)
    model = BERTClassifier(config, parameters)
    model.LoadParameters()
    datas,labels = GetTrainDataset(DATASET_PATH,MAX_LEN,SPLIT_RATE,BATCH_SIZE)
    model = Train(model,datas, labels, LR, NUM_EPOCHS,MODEL_SAVE_PATH)
    testDatas,testLabels = GetTestDataset(DATASET_PATH,MAX_LEN,SPLIT_RATE,BATCH_SIZE)
    Inference(model,datas, labels)
    SaveModel(model, MODEL_SAVE_PATH)
    newModel = LoadModel(MODEL_SAVE_PATH)
    Inference(newModel,datas, labels)

if __name__ == "__main__":
    MultiTest()
    #DataFlowTest()
    #SingleTest()
    #OnlyInference()
    #main()
    #WriteTfLite(MODEL_SAVE_PATH, TFLITE_PATH)
    #WriteInt8TFLite(MODEL_SAVE_PATH, TFLITE_INT8_PATH)
    #TfliteTest()