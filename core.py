import json
import os

import cv2
import skimage
import imagesize

import numpy as np

import mrcnn.visualize
from mrcnn import config, utils

CLASS_NAMES = ['BG', 'knife', 'pistol', 'carabine', 'rifle']


def random_colors():
    return mrcnn.visualize.random_colors(len(CLASS_NAMES), bright=True)


def rgb_percents_to_rgb(percent_rgb_color):
    return [int(c * 255) for c in percent_rgb_color]


def rgb_percents_to_bgr(percent_rgb_color):
    return rgb_percents_to_rgb(percent_rgb_color)[:: -1]


def visualize(frame, boxes, masks, class_ids, scores, show_mask=True, colors=None, captions=None):
    boxes_count = boxes.shape[0]
    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # Generate random colors
    if colors is None:
        colors = random_colors()

    masked_frame = frame.copy()
    for i in range(boxes_count):
        y1, x1, y2, x2 = boxes[i]

        # Label
        if not captions:
            score = scores[i] if scores is not None else None
            label = CLASS_NAMES[class_ids[i]]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]

        rgb_color_percents = colors[i]
        rgb_color = rgb_percents_to_rgb(rgb_color_percents)

        white_rgb_color = (255, 255, 255)

        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.7

        text_thickness = 1
        box_thickness = 2

        if show_mask:
            masked_frame = mrcnn.visualize.apply_mask(masked_frame, masks[:, :, i], rgb_color_percents, alpha=0.6)

        cv2.rectangle(masked_frame, (x1, y1), (x2, y2), rgb_color, thickness=box_thickness, lineType=cv2.LINE_AA)

        y1_n = (y1 - 10 if y1 > 0 else y1 + 20)
        x1_n = x1 + 5
        cv2.putText(masked_frame, caption, (x1_n, y1_n), font_face, font_scale, white_rgb_color,
                    thickness=text_thickness)

    return masked_frame


def train(model, dataset_path, epochs):
    # Training dataset.
    dataset_train = TrainDataset()
    dataset_train.load_custom(dataset_path, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = TrainDataset()
    dataset_val.load_custom(dataset_path, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=model.config.LEARNING_RATE,
                epochs=epochs,
                layers='heads')


class InferenceConfig09(config.Config):
    NAME = "object"

    # Number of classes (including background)
    NUM_CLASSES = len(CLASS_NAMES)

    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.9


class InferenceConfig07(config.Config):
    NAME = "object"

    # Number of classes (including background)
    NUM_CLASSES = len(CLASS_NAMES)

    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.7


class InferenceConfig08(config.Config):
    NAME = "object"

    # Number of classes (including background)
    NUM_CLASSES = len(CLASS_NAMES)

    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.8


class InferenceConfigBatch2(config.Config):
    NAME = "object"

    # Number of classes (including background)
    NUM_CLASSES = len(CLASS_NAMES)

    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    DETECTION_MIN_CONFIDENCE = 0.9


class InferenceConfigBatch3(config.Config):
    NAME = "object"

    # Number of classes (including background)
    NUM_CLASSES = len(CLASS_NAMES)

    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 3
    DETECTION_MIN_CONFIDENCE = 0.9


class TrainConfig(config.Config):
    NAME = "object"

    # Number of classes (including background)
    NUM_CLASSES = len(CLASS_NAMES)

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9


class TrainDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Load a subset of the Dog-Cat dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        class_names = CLASS_NAMES[1:]

        # Add classes. We have only one class to add.
        for i, c in enumerate(class_names, 1):
            self.add_class("object", i, c)

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }

        # We mostly care about the x and y coordinates of each region
        annotations_json = json.load(open(os.path.join(dataset_dir, "annotations.json")))
        annotations = list(annotations_json.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        # annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # print(a)
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions']] if 'regions' in a else []
            objects = [s['region_attributes']['type'] for s in a['regions']] if 'regions' in a else []

            name_dict = {c[1]: c[0] for c in enumerate(class_names, 1)}
            num_ids = [name_dict[a] for a in objects]

            image_path = os.path.join(dataset_dir, a['filename'])
            print("image path: {}".format(image_path))
            print("objects: {}".format(objects))

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            width, height = imagesize.get(image_path)

            self.add_image(
                "object",  # for a single class just add the name here
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids  # np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
