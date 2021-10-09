import argparse
import os
import sys

import cv2

import mrcnn.model as modellib
import test_cv_core as core


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Root directory of the project
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)  # To find local version of the library

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True, help="relative path to model weights file")
ap.add_argument("-i", "--in", required=True, help="relative path to input image")
ap.add_argument("-o", "--out", required=False, help="relative path to output image (path should be exists)")
args = vars(ap.parse_args())

# Path to weights file
WEIGHTS_PATH = os.path.join(ROOT_DIR, *os.path.split(args["weights"]))
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(WEIGHTS_PATH)

# Path to image
IN_PATH = os.path.join(ROOT_DIR, *os.path.split(args["in"]))
if not os.path.exists(IN_PATH):
    raise FileNotFoundError(IN_PATH)

WITH_OUTPUT = args["out"] is not None
OUT_PATH = os.path.join(ROOT_DIR, *os.path.split(args["out"])) if WITH_OUTPUT else None


# Create model in inference mode
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=core.InferenceConfig())

model.load_weights(WEIGHTS_PATH, by_name=True)
print("Weights loaded, path:", WEIGHTS_PATH)

# Load image
image = cv2.imread(IN_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("Image loaded, path:", IN_PATH)

result = model.detect([image_rgb], verbose=1)[0]

# Visualize results
result_image = core.visualize(image, result['rois'], result['masks'], result['class_ids'], result['scores'])

cv2.imshow("MRCNN Image", result_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# If there is output file try to save result
if WITH_OUTPUT:
    if cv2.imwrite(OUT_PATH, result_image):
        print("Image saved, path:", OUT_PATH)
    else:
        print("Failed to save image, path:", OUT_PATH, file=sys.stderr)
