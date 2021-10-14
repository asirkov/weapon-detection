import cv2

import mrcnn.visualize
from mrcnn import config


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
        cv2.putText(masked_frame, caption, (x1_n, y1_n), font_face, font_scale, white_rgb_color, thickness=text_thickness)

    return masked_frame


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
