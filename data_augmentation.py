"""
This script is an automated tool for data augmentation.
It mainly deals with the images in folder: ./dd2419_coco

Download dependencies:
pip install albumentations -U
"""

import os
import json
import random
from abc import ABC, abstractmethod

import cv2
import albumentations as A

# NOTE: using 'pip install albumentations -U' may not download the latest version?
# How to import `Rotate` depends on the version.
from albumentations.augmentations.transforms import Rotate

from config import NUM_CATEGORIES, CATEGORY_DICT, IMG_W, IMG_H


# data augmentation config
MIN_VIS = 0.8
DUP_TIMES = 20  # duplicate how many images
ANN_PATH = "dd2419_coco/annotations/training.json"
IMG_PATH = "dd2419_coco/training/"
NEW_ANN_PATH = None  # TODO: save json
NEW_IMG_PATH = "dd2419_coco/training_new/"
BBOX_PARAM = A.BboxParams(
    format="coco", min_visibility=MIN_VIS, label_fields=["class_labels"]
)
# `min_visibility` ensures that the bbox is not lost.
# `label_fields` will be an argument when using this transform


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
            Rotate(limit=(-10, 10), p=0.2),
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
        super.__init__(strategy_name)
        self.__transforms = [
            A.RandomSizedBBoxSafeCrop(width=IMG_W, height=IMG_H, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            Rotate(limit=(-10, 10), p=0.3),
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


class DataAugmentation:
    """
    A class for image data augmentation.
    """

    def __init__(self, dup_times=DUP_TIMES):
        self.images = []
        self.data = {}
        self.annotations = []
        self.dup_times = dup_times

    def read(self, path):
        """
        Read images and annotations.
        """
        with open(path) as json_file:
            self.data = json.load(json_file)
            self.annotations = self.data["annotations"]

    @staticmethod
    def label2strategy(label):
        """
        Args:
            label: int
        Returns: a Strategy object
        """
        if label in [2, 11, 12]:
            # TODO: more labels?
            return NoFlip()
        else:
            return AllTransform()

    @staticmethod
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

    def augment(self, debug=False):

        for idx in range(len(self.annotations)):
            idx_str = "%06d" % idx  # pad zero
            img_path = IMG_PATH + idx_str + ".jpg"
            image = cv2.imread(img_path)
            bbox = self.annotations[idx]["bbox"]
            label = self.annotations[idx]["category_id"]
            # label_str = str(label)

            for _ in range(self.dup_times):
                strategy = DataAugmentation.label2strategy(label)
                transform = strategy.get_transform()

                # use pipeline to transform
                transformed = transform(
                    image=image, bboxes=[bbox], class_labels=[label]
                )
                image_transformed = transformed["image"]  # ndarray, (W, H, C=3)
                bbox_transformed = transformed["bboxes"]
                label_transformed = transformed["class_labels"]
                self.images.append(image_transformed)
                # TODO: generate JSON file

            if debug:
                DataAugmentation.show_image_array(
                    image_transformed.copy(), bbox_transformed, label_transformed
                )
                break

    def write(self):
        """
        Save images.
        """
        object_dir = os.getcwd() + "/" + NEW_IMG_PATH
        if not os.path.exists(object_dir):
            os.makedirs("/" + NEW_IMG_PATH)
        for idx, img in enumerate(self.images):
            save_path = NEW_IMG_PATH + str(idx) + ".jpg"
            cv2.imwrite(save_path, img)


def data_augmentation(debug=False):
    da = DataAugmentation()
    da.read(ANN_PATH)
    da.augment(debug)
    da.write()


if __name__ == "__main__":
    data_augmentation(debug=True)
