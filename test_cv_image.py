import argparse
import os
import sys

import cv2
import tensorflow as tf

import mrcnn.model as modellib
from mrcnn import config
from mrcnn.visualize import random_colors, apply_mask


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
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
        cv2.putText(masked_frame, caption, (x1, (y1 - 10 if y1 > 0 else y1 + 20)), font_name, font_size, rgb_color)

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
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=InferenceConfig())

model.load_weights(WEIGHTS_PATH, by_name=True)
print("Weights loaded, path:", WEIGHTS_PATH)

# Load image
image = cv2.imread(IN_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("Image loaded, path:", IN_PATH)

result = model.detect([image_rgb], verbose=1)[0]

# Visualize results
result_image = visualize(image, result['rois'], result['masks'], result['class_ids'], CLASS_NAMES, result['scores'])

cv2.imshow("MRCNN Image", result_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

if WITH_OUTPUT:
    if cv2.imwrite(OUT_PATH, result_image):
        print("Image saved, path:", OUT_PATH)
    else:
        print("Failed to save image, path:", OUT_PATH)
