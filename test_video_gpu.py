import argparse
import glob
import os
import shutil
import sys
from datetime import datetime

import cv2
import tensorflow as tf

import mrcnn.model as modellib
import core

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
ap.add_argument("-o", "--out", required=True, help="relative path to output video")
ap.add_argument("-s", "--skip", required=False, default=3, help="step for frames skip, should be <= 0")
args = vars(ap.parse_args())

# Path to weights file
WEIGHTS_PATH = os.path.join(ROOT_DIR, *os.path.split(args["weights"]))
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(WEIGHTS_PATH)

# Path to input
IN_PATH = os.path.join(ROOT_DIR, *os.path.split(args["in"]))
if not os.path.exists(IN_PATH):
    raise FileNotFoundError(IN_PATH)

# Path to output
OUT_PATH = os.path.join(ROOT_DIR, *os.path.split(args["out"]))

TMP_DIR = os.path.join(ROOT_DIR, "tmp")
# Clear /tmp directory
if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)

os.mkdir(TMP_DIR)

SKIP = args["skip"]

# Create model in inference mode
model = modellib.MaskRCNN(mode="inference", model_dir=WEIGHTS_PATH, config=core.InferenceConfig08())

model.load_weights(WEIGHTS_PATH, by_name=True)
print("Weights loaded, path: {}".format(WEIGHTS_PATH))

# Load video
cap = cv2.VideoCapture(IN_PATH)
print("Video loaded, path: {}".format(IN_PATH))


COLORS = core.random_colors()

IN_FRAME_RATE = cap.get(cv2.CAP_PROP_FPS)

FRAME_SHAPE = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
FRAMES_COUNT = cap.get(cv2.CAP_PROP_FRAME_COUNT)

FRAME_NUMBER = 0

start_time = datetime.now()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if SKIP <= 0 or FRAME_NUMBER % SKIP == 0:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = model.detect([frame_rgb], verbose=0)[0]

        # Visualize results
        result_frame = core.visualize(frame,
                                      result['rois'],
                                      result['masks'],
                                      result['class_ids'],
                                      result['scores'],
                                      colors=COLORS)

        filename = os.path.join(TMP_DIR, "{}.jpg".format(FRAME_NUMBER))
        if cv2.imwrite(filename, result_frame):
            print("Frame saved, path: {}".format(filename))
        else:
            print("Failed to save frame, path: {}".format(filename))
            exit(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    print("Frames proceed: {}/{}".format(FRAME_NUMBER, int(FRAMES_COUNT)))
    FRAME_NUMBER += 1

end_time = datetime.now() - start_time

print("Input Video duration: {} seconds".format(int(FRAMES_COUNT / IN_FRAME_RATE)))
print("Completed in {} seconds".format(int(end_time.seconds)))

cap.release()

OUT_FRAME_RATE = int(IN_FRAME_RATE / SKIP if SKIP > 0 else IN_FRAME_RATE)
out = cv2.VideoWriter(OUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), OUT_FRAME_RATE, frameSize=FRAME_SHAPE)

for filename in sorted(glob.glob(os.path.join(TMP_DIR, "*.jpg")),
                       key=lambda f: int(os.path.split(f)[-1].split(".")[0])):
    frame = cv2.imread(filename)
    out.write(frame)

out.release()
print("Output Video saved, path: {}".format(OUT_PATH))

shutil.rmtree(TMP_DIR)

cap = cv2.VideoCapture(OUT_PATH)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("RCNN Video", frame)

cap.release()

cv2.destroyAllWindows()
