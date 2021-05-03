import cv2
from config import CATEGORY_DICT
import json

"""
Use this script to test data integration and data augmentation work well.
"""


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


# IMG_PATH = "C:\\Users\\Jerry TAN\\Downloads\\Integrate_data_collection\\training_images\\01456\\"
# ANN_PATH = "./jsons/training_all.json"

IMG_PATH = "./dd2419_coco/01456_all/"
ANN_PATH = "./jsons/training_all_aug.json"

# IMG_PATH = "./dd2419_coco/01456/"
# ANN_PATH = "./jsons/training_all.json"

if __name__ == "__main__":
    with open(ANN_PATH) as json_file:
        data = json.load(json_file)
        images = data["images"]
        annotations = data["annotations"]
    #     bboxs = list(map(lambda x: x["bbox"], annotations))

    for i in range(1, 10000, 110):
        ann_id = i

        try:
            img_id = annotations[ann_id]["image_id"]
        except:
            print("test ends")
            break

        img_path = IMG_PATH + images[img_id]["file_name"]
        print(img_path)
        image = cv2.imread(img_path)
        bbox = annotations[ann_id]["bbox"]
        label = annotations[ann_id]["category_id"]

        show_image_array(image, bboxes=bbox, labels=label)

