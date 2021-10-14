import argparse
import os
import sys

import tensorflow as tf

import core
from mrcnn import model as modellib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

DEVICE = "/device:GPU:0"  # /CPU:0 or /device:GPU:0

device_name = tf.test.gpu_device_name()
if device_name != DEVICE:
    raise SystemError('GPU device not found')

print('Found GPU at: {}'.format(device_name))

# Root directory of the project
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)  # To find local version of the library

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True, help="relative path to model weights file")
ap.add_argument("-d", "--dataset", required=True, help="relative path to dataset")
args = vars(ap.parse_args())

# Path to weights file
WEIGHTS_PATH = os.path.join(ROOT_DIR, *os.path.split(args["weights"]))
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(WEIGHTS_PATH)

# Path to dataset
DATASET_PATH = os.path.join(ROOT_DIR, *os.path.split(args["dataset"]))
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(DATASET_PATH)

# Directory to save logs and model checkpoints, if not provided
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Create model
model = modellib.MaskRCNN(mode="training", config=core.TrainConfig(), model_dir=LOGS_DIR)

model.load_weights(WEIGHTS_PATH, by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

print("Weights loaded, path:", WEIGHTS_PATH)

# Start training
EPOCHS = 10
print("Training started, dataset path: {}, epochs: {}".format(DATASET_PATH, EPOCHS))
core.train(model, DATASET_PATH, EPOCHS)

print("Training completed, {} epochs passed, logs path: {}".format(EPOCHS, LOGS_DIR))
