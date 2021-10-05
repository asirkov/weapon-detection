import os
import argparse
import os
import sys

import cv2
import matplotlib as plt
import matplotlib.patches as patches
import numpy as np
import tensorflow as tf
from skimage.measure import find_contours

import mrcnn.model as modellib
from mrcnn import config, visualize
from mrcnn.visualize import random_colors, apply_mask

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


# def display_instances(image, boxes, masks, class_ids, class_names,
#                       scores=None,
#                       title="",
#                       ax=None,
#                       show_mask=True,
#                       show_bbox=True,
#                       colors=None,
#                       captions=None):
#     # Number of instances
#     N = boxes.shape[0]
#     if not N:
#         print("\n*** No instances to display *** \n")
#     else:
#         assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
#
#     # Generate random colors
#     colors = colors or random_colors(N)
#
#     # Show area outside image boundaries.
#     cv2.imshow(title, image)
#
#     masked_image = image.astype(np.uint32).copy()
#     for i in range(N):
#         color = colors[i]
#
#         # Bounding box
#         if not np.any(boxes[i]):
#             # Skip this instance. Has no bbox. Likely lost in image cropping.
#             continue
#         y1, x1, y2, x2 = boxes[i]
#         if show_bbox:
#             p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
#                                   alpha=0.7, linestyle="solid",
#                                   edgecolor=color, facecolor='none')
#             ax.add_patch(p)
#
#         # Label
#         if not captions:
#             class_id = class_ids[i]
#             score = scores[i] if scores is not None else None
#             label = class_names[class_id]
#             caption = "{} {:.3f}".format(label, score) if score else label
#         else:
#             caption = captions[i]
#         ax.text(x1, y1 + 8, caption,
#                 color='w', size=11, backgroundcolor="none")
#
#         # Mask
#         mask = masks[:, :, i]
#         if show_mask:
#             masked_image = apply_mask(masked_image, mask, color)
#
#         # Mask Polygon
#         # Pad to ensure proper polygons for masks that touch image edges.
#         padded_mask = np.zeros(
#             (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
#         padded_mask[1:-1, 1:-1] = mask
#         contours = find_contours(padded_mask, 0.5)
#         for verts in contours:
#             # Subtract the padding and flip (y, x) to (x, y)
#             verts = np.fliplr(verts) - 1
#             p = patches.Polygon(verts, facecolor="none", edgecolor=color)
#             ax.add_patch(p)
#
#     cv2.imshow(title, masked_image.astype(np.uint8))


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
ax = visualize.display_instances(image_rgb,
                  result['rois'],
                  result['masks'],
                  result['class_ids'],
                  CLASS_NAMES,
                  result['scores'])

cv2.waitKey(0)
cv2.destroyAllWindows()
