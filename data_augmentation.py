"""
This script is an automated tool for data augmentation.
It mainly deals with the images in folder: ./dd2419_coco

Download dependencies:
pip install albumentations -U

Tested with Python 3.6.9 and PyTorch 1.4.0
"""

import os
import json
import random
from abc import ABC, abstractmethod
from collections import Counter
from copy import deepcopy

import cv2
import albumentations as A

# NOTE: using 'pip install albumentations -U' may not download the latest version?
# How to import `Rotate` depends on the version.
from albumentations.augmentations.transforms import Rotate

from config import CATEGORY_DICT, IMG_W, IMG_H


# check data set before augmentation!
# IMAGE_ID_START = 1258  # beginning image_id of new images
# ID_START = 1257  # beginning id of new annotations
# 2008? 2286!
IMAGE_ID_START = 2286  # beginning image_id of new images
ID_START = 2286  # beginning id of new annotations

# data augmentation config
MIN_VIS = 0.99
# DUP_TIMES = 20  # duplicate and transform how many times for each image
DUP_TIMES = {
    0: 0,
    2: 0,
    10: 1,
    13: 0,
    12: 0,
    11: 0,
    1: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    8: 1,
    9: 1,
    14: 1,
    7: 8,
}

# Counter({0: 287,
#          2: 246,
#          10: 125,       X2
#          13: 201,
#          12: 198,
#          11: 200,
#          1: 119,   X2
#          3: 122,
#          4: 129,
#          5: 121,
#          6: 128,
#          7: 30,
#          8: 126,
#          9: 127,
#          14: 126})


BBOX_PARAM = A.BboxParams(
    format="coco", min_visibility=MIN_VIS, label_fields=["class_labels"]
)
# NOTE: `min_visibility` ensures that the bbox is not lost.
# `label_fields` will be an argument when using this transform

# overwrite the original file or not
OVERWRITE = False   # TODO: if testing, set this FALSE
# set reading path
# ANN_PATH = "dd2419_coco/annotations/training_all_new.json"
ANN_PATH = "jsons/training_all.json"
IMG_PATH = "dd2419_coco/01456/"
# get writing path
if OVERWRITE:
    # original images and annotations will be overwritten
    # recommend that make a backup of original data set
    NEW_ANN_PATH = ANN_PATH
    NEW_IMG_PATH = IMG_PATH
else:
    # new images and json file are generated in a new path
    # NEW_ANN_PATH = "dd2419_coco/annotations/training_all_new_aug.json"
    NEW_ANN_PATH = "jsons/training_all_aug.json"
    # NEW_IMG_PATH = (
    #     "dd2419_coco/training_all_images_new/"
    #     # save in a new folder and will merge later
    # )
    NEW_IMG_PATH = (
        "dd2419_coco/01456_new/"
        # save in a new folder and will merge later
    )


class Strategy(ABC):
    """
    Strategies for different combinations of transforms.
    Each strategy contains a transform.
    If a new strategy is needed, a derived class should be implemented.
    """

    def __init__(self, strategy_name):
        self.strategy_name = strategy_name

    @abstractmethod
    def get_transform(self):
        pass


class AllTransform(Strategy):
    """
    A derived class of `Strategy`, performing all kinds of transforms.
    """

    def __init__(self, strategy_name="AllTransform"):
        super().__init__(strategy_name)
        self.__transforms = [
            A.RandomSizedBBoxSafeCrop(width=IMG_W, height=IMG_H, p=0.5),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            Rotate(limit=(-4, 4), p=0.2),
            A.Blur(p=0.2),
            A.ColorJitter(p=0.2),
            A.GaussNoise(p=0.2),
        ]
        random.shuffle(self.__transforms)

    def get_transform(self):
        return A.Compose(
            self.__transforms,
            bbox_params=BBOX_PARAM,
        )


class NoFlip(Strategy):
    """
    A derived class of `Strategy`, whose object does not flip images.
    """

    def __init__(self, strategy_name="NoFlip"):
        super().__init__(strategy_name)
        self.__transforms = [
            A.RandomSizedBBoxSafeCrop(width=IMG_W, height=IMG_H, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            Rotate(limit=(-4, 4), p=0.3),
            A.Blur(p=0.2),
            A.ColorJitter(p=0.3),
            A.GaussNoise(p=0.2),
        ]
        random.shuffle(self.__transforms)

    def get_transform(self):
        return A.Compose(
            self.__transforms,
            bbox_params=BBOX_PARAM,
        )


# not as static method now
def label2strategy(label):
    """
    Get a Strategy object depending on the given label.
    Args:
        label: int
    Returns: a Strategy object
    """
    if label in [2, 3, 4, 5, 11, 12]:
        return NoFlip()
    else:
        return AllTransform()


def show_image_array(img, bboxes=None, labels=None):
    """
    Show image with the bounding box.
    Args:
        img: numpy.ndarray
        bboxes: list of tuples
        labels: list of int
    """

    # add bounding box or not
    if bboxes is not None and labels is not None:
        # single input case
        if type(labels) is int:
            labels = [labels]
            bboxes = [bboxes]

        # the length should match
        assert len(bboxes) == len(labels)

        for idx in range(len(bboxes)):
            bbox = bboxes[idx]
            lbl = labels[idx]
            bbox = list(map(int, bbox))
            start_point = (bbox[0], bbox[1])
            text_point = (bbox[0], bbox[1] - 5)
            end_point = (bbox[0] + bbox[2], bbox[1] + bbox[3])
            cv2.rectangle(img, start_point, end_point, (0, 0, 255), 2)
            cv2.putText(
                img,
                CATEGORY_DICT[lbl]["name"],
                text_point,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

    cv2.imshow("show image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # There is a big problem that RAM is not enough for too many images.
    # set `DEBUG` to zero or negative to exit debug mode.
    DEBUG = 0
    new_image_id, new_id = IMAGE_ID_START, ID_START

    # save images path
    object_dir = os.getcwd() + "/" + NEW_IMG_PATH
    if not os.path.exists(object_dir):
        os.makedirs("./" + NEW_IMG_PATH)

    # read old json
    with open(ANN_PATH) as json_file:
        data = json.load(json_file)
        images = data["images"]
        annotations = data["annotations"]
        img_json = deepcopy(images)
        ann_json = deepcopy(annotations)

    # augment
    for idx in range(len(annotations)):
        if idx % 100 == 0:
            print('begin ann ', idx, '...')

        # image_name = data["images"][idx]["file_name"]
        image_name = images[annotations[idx]["image_id"]]["file_name"]
        img_path = IMG_PATH + image_name
        image = cv2.imread(img_path)
        bbox = annotations[idx]["bbox"]
        label = annotations[idx]["category_id"]

        for _ in range(DUP_TIMES[label]):
            strategy = label2strategy(label)
            transform = strategy.get_transform()

            # show_image_array(
            #     image.copy(), bbox, label
            # )
            #
            # # ............debug
            # input("Press Enter to continue...")
            # # ............

            # use pipeline to transform
            try:
                transformed = transform(
                    image=image, bboxes=[bbox], class_labels=[label]
                )
            except:
                continue

            image_transformed = transformed["image"]  # ndarray, (W, H, C=3)
            bbox_transformed = transformed["bboxes"]
            label_transformed = transformed["class_labels"]

            # show_image_array(
            #     image_transformed.copy(), bbox_transformed, label_transformed
            # )

            # check if bounding box exists
            if label_transformed:
                save_name = "AUG_" + str(new_image_id) + ".jpg"
                save_path = NEW_IMG_PATH + save_name
                cv2.imwrite(save_path, image_transformed)
                img_json.append({
                        "id": new_image_id,
                        "width": 640,
                        "height": 480,
                        "file_name": save_name,
                    })
                ann_json.append({
                        "id": new_id,
                        "image_id": new_image_id,
                        "category_id": label_transformed[0],
                        "bbox": list(bbox_transformed[0]),
                    })

                new_image_id, new_id = new_image_id + 1, new_id + 1

        # --- loop end --

    # update json
    new_data = dict()
    new_data["info"] = data["info"]
    new_data["images"] = img_json
    new_data["annotations"] = ann_json
    new_data["categories"] = data["categories"]

    with open(NEW_ANN_PATH, "w") as json_file:
        json.dump(new_data, json_file, indent=2)
        print("JSON file written")
