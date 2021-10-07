import cv2

from mrcnn import config
from mrcnn.visualize import random_colors, apply_mask

CLASS_NAMES = ['BG', 'knife', 'pistol', 'carabine']


def get_colors():
    return random_colors(len(CLASS_NAMES))


def visualize(frame, boxes, masks, class_ids, scores, show_mask=True, colors=None, captions=None):
    boxes_count = boxes.shape[0]
    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # Generate random colors
    colors = colors or random_colors(len(CLASS_NAMES))

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
    DETECTION_MIN_CONFIDENCE = 0.7


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

