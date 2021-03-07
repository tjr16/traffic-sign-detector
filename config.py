"""
Define some parameters in this script.
"""
NUM_CATEGORIES = 15

CATEGORY_DICT = {
    0: {"name": "no_bicycle"},
    1: {"name": "__CLASS 1__"},
    2: {"name": "dangerous_left"},
    3: {"name": "__CLASS 3__"},
    4: {"name": "__CLASS 4__"},
    5: {"name": "__CLASS 5__"},
    6: {"name": "__CLASS 6__"},
    7: {"name": "__CLASS 7__"},
    8: {"name": "__CLASS 8__"},
    9: {"name": "__CLASS 9__"},
    10: {"name": "residential"},
    11: {"name": "narrows_from_left"},
    12: {"name": "narrows_from_right"},
    13: {"name": "roundabout"},
    14: {"name": "__CLASS 14__"}
}

BATCH_SIZE = 8
CONF_THRESHOLD = 0.8    # threshold of confidence
IOU_THRESHOLD = 0.4  # threshold of IOU for NMS
OUTPUT_FUNC = 'softmax'
LEARNING_RATE = 1e-4
