from ..preprocess.data import GetTestDataset
import tensorflow as tf 
from progressbar import ProgressBar
from dotenv import load_dotenv
import numpy as np
load_dotenv()
import os 
import time

TFLITE_PATH = os.getenv('TFLITE_PATH')
TFLITE_INT8_PATH = os.getenv('TFLITE_INT8_PATH')
DATASET_PATH = os.getenv('DATASET_PATH')
MAX_LEN = int(os.getenv('MAX_LEN'))
SPLIT_RATE = float(os.getenv('SPLIT_RATE'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))

def TfliteTest():
    print("Tflite Inferencing...")
    interpreter = GetInterpreter(TFLITE_INT8_PATH)
    metrics = tf.metrics.SparseCategoricalAccuracy()
    batchDatas, batchLabels = GetTestDataset(DATASET_PATH,MAX_LEN,SPLIT_RATE,BATCH_SIZE)
    now = 0
    total = len(batchDatas)
    start = time.time()
    pBar = ProgressBar().start()
    for data,label in zip(batchDatas,batchLabels):
        y_pred = TfliteInference(interpreter, data, label)
        metrics.update_state(label,y_pred)
        pBar.update(int((now / (total - 1)) * 100))
        now += 1
    pBar.finish()
    print(f'cost time: {round(time.time() - start,3)} sec')
    print(f'Lite accuracy:{metrics.result().numpy()}')

def TfliteInference(interpreter,datas,labels):
    inputs = interpreter.get_input_details()[0]['index']
    outputs = interpreter.get_output_details()[0]['index']
    interpreter.set_tensor(inputs, datas)
    interpreter.invoke()
    return interpreter.get_tensor(outputs)
    
def RandomTest(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.int32)
    print(input_data)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
    print(tflite_model_predictions)

def GetInterpreter(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.resize_tensor_input(0, [BATCH_SIZE, MAX_LEN])
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Input Shape:",input_details[0]['shape'])
    print("Input Type:",input_details[0]['dtype'])
    print("Output Shape:",output_details[0]['shape'])
    print("Output Type:",output_details[0]['dtype'])
    return interpreter


if __name__ == "__main__":
    pass