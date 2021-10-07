import argparse
import os
import sys

import cv2
import tensorflow as tf

import mrcnn.model as modellib

import test_cv_core as core


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

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True, help="relative path to model weights file")
ap.add_argument("-i", "--in", required=True, help="relative path to input video")
args = vars(ap.parse_args())

# Path to weights file
WEIGHTS_PATH = os.path.join(ROOT_DIR, *os.path.split(args["weights"]))
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(WEIGHTS_PATH)

# Path to video
IN_PATH = os.path.join(ROOT_DIR, *os.path.split(args["in"]))
if not os.path.exists(IN_PATH):
    raise FileNotFoundError(IN_PATH)


# Create model in inference mode
model = modellib.MaskRCNN(mode="inference", model_dir=WEIGHTS_PATH, config=core.InferenceConfig08())

model.load_weights(WEIGHTS_PATH, by_name=True)
print("Weights loaded, path:", WEIGHTS_PATH)

# Load video
cap = cv2.VideoCapture(IN_PATH)
print("Video loaded, path:", IN_PATH)


COLORS = core.get_colors()

FRAME_NUMBER = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if FRAME_NUMBER % 5 == 0:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = model.detect([frame_rgb], verbose=0)[0]

        # Visualize results
        result_frame = core.visualize(frame,
                                      result['rois'],
                                      result['masks'],
                                      result['class_ids'],
                                      result['scores'],
                                      colors=COLORS)

        scaled_result_frame = cv2.resize(result_frame, (640, 480))
        cv2.imshow('RCNN Video', scaled_result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    FRAME_NUMBER += 1

cap.release()
cv2.destroyAllWindows()
