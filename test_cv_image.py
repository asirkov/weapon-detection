import argparse
import os
import sys

import cv2
import numpy as np
import tensorflow as tf

import mrcnn.model as modellib
from mrcnn import config
from mrcnn.visualize import random_colors, apply_mask, display_instances

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Root directory of the project
ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR)  # To find local version of the library

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True, help="relative path to model weights file")
ap.add_argument("-i", "--image", required=True, help="relative path to input image")
args = vars(ap.parse_args())

# Path to weights file
WEIGHTS_PATH = os.path.join(ROOT_DIR, *os.path.split(args["weights"]))
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(WEIGHTS_PATH)

# Path to image
IMAGE_PATH = os.path.join(ROOT_DIR, *os.path.split(args["image"]))
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(IMAGE_PATH)

TEST_MODE = "inference"

# Device to load the neural network on.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

CLASS_NAMES = ['BG', 'knife', 'pistol', 'carabine']


def visualize(frame, boxes, masks, class_ids, class_names, scores,
              show_mask=True, colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask: To show masks and bounding boxes or not
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    boxes_count = boxes.shape[0]
    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # Generate random colors
    colors = colors or random_colors(len(class_names))

    masked_frame = frame.copy()
    for i in range(boxes_count):
        y1, x1, y2, x2 = boxes[i]

        # Label
        if not captions:
            score = scores[i] if scores is not None else None
            label = class_names[class_ids[i]]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]

        color = colors[i]
        font_name = cv2.FONT_HERSHEY_DUPLEX
        font_size = 0.7
        thickness = 2

        if show_mask:
            masked_frame = apply_mask(masked_frame, masks[:, :, i], color, alpha=0.6)

        rgb_color = [int(c) * 255 for c in color]

        cv2.rectangle(masked_frame, (x1, y1), (x2, y2), rgb_color, thickness)
        cv2.putText(masked_frame, caption, (x1, y1), font_name, font_size, rgb_color)

    return masked_frame


class InferenceConfig(config.Config):
    NAME = "object"

    # Number of classes (including background)
    NUM_CLASSES = len(CLASS_NAMES)

    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.9


# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode=TEST_MODE, model_dir=MODEL_DIR, config=InferenceConfig())
    model.load_weights(WEIGHTS_PATH, by_name=True)
    print("Weights loaded, path:", WEIGHTS_PATH)

# Load image
image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("Image loaded, path:", IMAGE_PATH)

result = model.detect([image_rgb], verbose=1)[0]

# Visualize results
result_image = visualize(image, result['rois'], result['masks'], result['class_ids'], CLASS_NAMES, result['scores'])

cv2.imshow("MRCNN Image", result_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
