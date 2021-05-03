"""Baseline detector model.

Inspired by
You only look once: Unified, real-time object detection, Redmon, 2016.
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from torchvision import ops
from config import *


class Detector(nn.Module):
    """Baseline module for object detection."""

    def __init__(self, num_categories=15, device='cuda'):
        """Create the module.

        Define all trainable layers.
        """
        super(Detector, self).__init__()

        self.num_categories = num_categories
        self.device = device

        self.features = models.mobilenet_v2(pretrained=True).features
        # output of mobilenet_v2 will be 1280x15x20 for 480x640 input images

        # for param in self.features[0: 14].parameters():
        #     param.requires_grad = False
        # the 15-19 layer is not fixed now.

        self.head = nn.Conv2d(
            in_channels=1280, out_channels=5+self.num_categories, kernel_size=1
        )
        # 1x1 Convolution to reduce channels to out_channels without changing H and W

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        # 1280x15x20 -> 5x15x20, where each element 5 channel tuple corresponds to
        #   (rel_x_offset, rel_y_offset, rel_x_width, rel_y_height, confidence
        # Where rel_x_offset, rel_y_offset is relative offset from cell_center
        # Where rel_x_width, rel_y_width is relative to image size
        # Where confidence is predicted IOU * probability of object center in this cell
        self.out_cells_x = 20.0
        self.out_cells_y = 15.0
        self.img_height = 480.0
        self.img_width = 640.0

    def forward(self, inp):
        """Forward pass.

        Compute output of neural network from input.
        """
        features = self.features(inp)   # batch, channel(1280), 15, 20

        # output: linear/ sigmoid/ 0-1 clamp
        out = self.head(features)   # batch, channel(20), 15, 20
        if OUTPUT_FUNC == 'sigmoid':
            out1 = self.sigmoid(out)
        elif OUTPUT_FUNC == 'clamp':
            out1 = torch.clamp(out, min=0, max=1)
        elif OUTPUT_FUNC == 'softmax':
            out1 = torch.zeros(out.size()).to(self.device)
            out1[:, :5, :, :] = self.sigmoid(out[:, :5, :, :])
            out1[:, 5:, :, :] = self.softmax(out[:, 5:, :, :])
        else:
            out1 = out

        return out1

    def b2v(self, coeffs, bb_index):
        """
        Convert a box parameters to its vertices.
        Args:
            coeffs: box coefficients, torch.Size([4])
            bb_index: where is the center located at?

        Returns:
            v : N x 2
        """
        width = self.img_width * coeffs[2]
        height = self.img_height * coeffs[3]
        xc = self.img_width / self.out_cells_x * (bb_index[1] + coeffs[0])
        yc = self.img_height / self.out_cells_y * (bb_index[0] + coeffs[1])
        xmin = xc - width / 2.0
        xmax = xc + width / 2.0
        ymin = yc - height / 2.0
        ymax = yc + height / 2.0
        return torch.Tensor([xmin, ymin, xmax, ymax]).reshape(1, -1)

    def decode_output(self, out, threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD):
        """Convert output to list of bounding boxes.

        Args:
            out (torch.tensor):
                The output of the network.
                Shape expected to be NxCxHxW with
                    N = batch size
                    C = channel size
                    H = image height
                    W = image width
                eg. torch.Size([8, 20, 15, 20])
            threshold (float):
                The threshold above which a bounding box will be accepted.
        Returns:
            bbs (List[List[Dict]]):
                List containing a list of detected bounding boxes in each image.
                Each dictionary contains the following keys:
                    - "x": Top-left corner column
                    - "y": Top-left corner row
                    - "width": Width of bounding box in pixel
                    - "height": Height of bounding box in pixel
                    - "category": Category
        """
        bbs = []

        # EACH IMAGE: decode bounding boxes
        for o in out:   # o: C x H x W (eg. 20 chan, 15 high, 20 wid)
            img_bbs = []    # bounding boxes for one image

            # find cells with bounding box center
            bb_indices = torch.nonzero(o[4, :, :] >= threshold)
            # n x 2, representing n centers

            # get box vertices and scores
            num_box = bb_indices.size()[0]
            all_boxes = torch.zeros(num_box, 4)
            all_score = torch.zeros(num_box)
            for idx, bb_index in enumerate(bb_indices):
                bb_coeffs = o[0:4, bb_index[0], bb_index[1]]
                all_boxes[idx, :] = self.b2v(bb_coeffs, bb_index)
                all_score[idx] = o[4, bb_index[0], bb_index[1]]

            # Non Maximum Suppression
            # return: the indices, torch.Size([n])
            reserved_indices = ops.nms(all_boxes, all_score, iou_threshold)
            bb_indices_new = bb_indices[reserved_indices, :]

            # loop over all cells with bounding box center
            for bb_index in bb_indices_new:
                bb_coeffs = o[0:4, bb_index[0], bb_index[1]]    # box param
                bb_conf = o[4, bb_index[0], bb_index[1]]    # confidence
                bb_cate = o[5:, bb_index[0], bb_index[1]]   # category

                # decode bounding box size and position
                width = self.img_width * bb_coeffs[2]
                height = self.img_height * bb_coeffs[3]
                y = (
                        self.img_height / self.out_cells_y * (bb_index[0] + bb_coeffs[1])
                        - height / 2.0
                )
                x = (
                        self.img_width / self.out_cells_x * (bb_index[1] + bb_coeffs[0])
                        - width / 2.0
                )
                category = torch.argmax(bb_cate, dim=0).item()  # Tensor to int

                img_bbs.append(
                    {
                        "width": width,
                        "height": height,
                        "x": x,
                        "y": y,
                        "category": category,
                        "confidence": bb_conf.item()
                    }
                )

            bbs.append(img_bbs)

        return bbs

    def input_transform(self, image, anns):
        """Prepare image and targets on loading.

        This function is called before an image is added to a batch.
        Must be passed as transforms function to dataset.

        Args:
            image (torch.Tensor):
                The image loaded from the dataset.
            anns (List):
                List of annotations in COCO format.
        Returns:
            Tuple:
                - (torch.Tensor) The image.
                - (torch.Tensor) The network target containing the bounding box.
        """
        # Convert PIL.Image to torch.Tensor
        image = transforms.ToTensor()(image)

        # Convert bounding boxes to target format

        # First two channels contain relativ x and y offset of bounding box center
        # Channel 3 & 4 contain relative width and height, respectively
        # Last channel is 1 for cell with bounding box center and 0 without

        # If there is no bb, the first 4 channels will not influence the loss
        # -> can be any number (will be kept at 0 zero)
        target = torch.zeros(5+self.num_categories, 15, 20)
        for ann in anns:
            x = ann["bbox"][0]
            y = ann["bbox"][1]
            width = ann["bbox"][2]
            height = ann["bbox"][3]
            category = ann["category_id"]

            x_center = x + width / 2.0
            y_center = y + height / 2.0
            x_center_rel = x_center / self.img_width * self.out_cells_x
            y_center_rel = y_center / self.img_height * self.out_cells_y
            x_ind = int(x_center_rel)
            y_ind = int(y_center_rel)
            x_cell_pos = x_center_rel - x_ind
            y_cell_pos = y_center_rel - y_ind
            rel_width = width / self.img_width
            rel_height = height / self.img_height

            # channels, rows (y cells), cols (x cells)
            target[4, y_ind, x_ind] = 1

            # bb size
            target[0, y_ind, x_ind] = x_cell_pos
            target[1, y_ind, x_ind] = y_cell_pos
            target[2, y_ind, x_ind] = rel_width
            target[3, y_ind, x_ind] = rel_height

            # categories
            target[5+category, y_ind, x_ind] = 1

        return image, target
