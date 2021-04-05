"""
Define some parameters in this script.
"""

# parameters of data set
NUM_CATEGORIES = 15
IMG_W = 640
IMG_H = 480

CATEGORY_DICT = {
    0: {"name": "no_bicycle"},
    1: {"name": "airport"},
    2: {"name": "dangerous_left"},
    3: {"name": "dangerous_right"},
    4: {"name": "follow_left"},
    5: {"name": "follow_right"},
    6: {"name": "junction"},
    7: {"name": "no_heavy_truck"},
    8: {"name": "no_parking"},
    9: {"name": "no_stopping_and_parking"},
    10: {"name": "residential"},
    11: {"name": "narrows_from_left"},
    12: {"name": "narrows_from_right"},
    13: {"name": "roundabout"},
    14: {"name": "stop"}
}

# weights of mse
WEIGHT_REG = 4
WEIGHT_NOOBJ = 0.5
WEIGHT_CLASS = 10

# parameters of network training
BATCH_SIZE = 8
CONF_THRESHOLD = 0.75    # threshold of confidence
IOU_THRESHOLD = 0.1  # threshold of IOU for NMS
OUTPUT_FUNC = 'sigmoid'
LEARNING_RATE = 1e-4
