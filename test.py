import json
import os
import sys
import random
import math
import re
import time
import numpy as np
import skimage
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

from mrcnn import utils
from mrcnn import config
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log


# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Path to trained weights
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
WEIGHTS_PATH = os.path.join(LOGS_DIR, "mask_rcnn_object_0010.h5")
# WEIGHTS_PATH = os.path.join(LOGS_DIR, "logs/mask_rcnn_coco.h5")

TEST_MODE = "inference"


class InferenceConfig(config.Config):
    NAME = "object"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # Background + knife + pistol + carabine

    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.9


config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# LOAD MODEL
# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode=TEST_MODE, model_dir=WEIGHTS_PATH, config=config)
    model.load_weights(WEIGHTS_PATH, by_name=True)

class_names = ['BG', 'knife', 'pistol', 'carabine']


# Load a random image from the images folder

TEST_DIR = os.path.join(ROOT_DIR, "test")

# ax = plt.figure(figsize=(12, 10))

for image_name in os.listdir(TEST_DIR):
    image_path = os.path.join(TEST_DIR, image_name)
    image = plt.imread(image_path)

    # ax = skimage.io.imshow(image)

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results

    for result in results:
        visualize.display_instances(image,
                                    result['rois'],
                                    result['masks'],
                                    result['class_ids'],
                                    class_names,
                                    result['scores'],
                                    title="Predictions",
                                    # ax=ax
                                    )
        # plt.draw()

        mask = result['masks']
        mask = mask.astype(int)

        print(mask.shape)


input()
