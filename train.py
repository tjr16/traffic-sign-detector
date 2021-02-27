"""Training script for detector."""
from __future__ import print_function

import argparse
from datetime import datetime
import os

import torch
from torch import nn
from torchvision.datasets import CocoDetection
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import wandb

import utils
from detector import Detector

# Some parameters
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
    9: {"name": "__CLASS 8__"},
    10: {"name": "residential"},
    11: {"name": "narrows_from_left"},
    12: {"name": "narrows_from_right"},
    13: {"name": "roundabout"},
    14: {"name": "__CLASS 14__"},
    15: {"name": "__CLASS 15__"}
}


def train(max_iter, device="cpu"):
    """Train the network.

    Args:
        device: The device to train on."""

    global NUM_CATEGORIES, CATEGORY_DICT

    wandb.init(project="detector_baseline")

    # Init model
    detector = Detector(NUM_CATEGORIES).to(device)

    wandb.watch(detector)

    dataset = CocoDetection(
        root="./dd2419_coco/training",
        annFile="./dd2419_coco/annotations/training.json",
        transforms=detector.input_transform,
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    # training params
    max_iterations = wandb.config.max_iterations = max_iter

    learning_rate = wandb.config.learning_rate = 1e-4
    weight_reg = wandb.config.weight_reg = 1
    weight_noobj = wandb.config.weight_noobj = 1

    # run name (to easily identify model later)
    time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    run_name = wandb.config.run_name = "saved_models/det_{}".format(time_string)

    # init optimizer
    optimizer = torch.optim.Adam(detector.parameters(), lr=learning_rate)

    # load test images
    # these will be evaluated in regular intervals
    test_images = []
    show_test_images = False
    directory = "./test_images"
    if not os.path.exists(directory):
        os.makedirs(directory)
    for file_name in os.listdir(directory):
        if file_name.endswith(".jpg"):
            file_path = os.path.join(directory, file_name)
            test_image = Image.open(file_path)
            test_images.append(TF.to_tensor(test_image))

    if test_images:
        test_images = torch.stack(test_images)
        test_images = test_images.to(device)
        show_test_images = True

    print("Training started...")

    current_iteration = 1
    while current_iteration <= max_iterations:
        for img_batch, target_batch in dataloader:
            img_batch = img_batch.to(device)    # torch.Size([8, 3, 480, 640])
            target_batch = target_batch.to(device)  # Batch, 5+category, 15, 20

            # run network
            out = detector(img_batch)   # torch.Size([8, 5+category, 15, 20])

            # positive / negative indices
            # (this could be passed from input_transform to avoid recomputation)
            # check tensor: torch.Size([8, 15, 20])
            pos_indices = torch.nonzero(target_batch[:, 4, :, :] == 1, as_tuple=True)
            # (torch.Size([8]),  torch.Size([8]),  torch.Size([8]))
            neg_indices = torch.nonzero(target_batch[:, 4, :, :] == 0, as_tuple=True)
            # (torch.Size([2392]), torch.Size([2392]), torch.Size([2392]))

            # compute loss
            # bounding box err
            reg_mse = nn.functional.mse_loss(
                out[pos_indices[0], 0:4, pos_indices[1], pos_indices[2]],   # torch.Size([8, 4])
                # each [4]: out[xx[0][0], 0:4, xx[1][0], xx[2][0]
                target_batch[pos_indices[0], 0:4, pos_indices[1], pos_indices[2]],  # torch.Size([8, 4])
            )
            # confidence err where box exists
            pos_mse = nn.functional.mse_loss(
                out[pos_indices[0], 4, pos_indices[1], pos_indices[2]],
                target_batch[pos_indices[0], 4, pos_indices[1], pos_indices[2]],
            )
            # confidence err where box not exists
            neg_mse = nn.functional.mse_loss(
                out[neg_indices[0], 4, neg_indices[1], neg_indices[2]],
                target_batch[neg_indices[0], 4, neg_indices[1], neg_indices[2]],
            )
            # class err
            cls_mse = nn.functional.mse_loss(
                out[pos_indices[0], 5:, pos_indices[1], pos_indices[2]],
                target_batch[pos_indices[0], 5:, pos_indices[1], pos_indices[2]]
            )
            loss = pos_mse + weight_reg * reg_mse + weight_noobj * neg_mse + cls_mse

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log(
                {
                    "total loss": loss.item(),
                    "loss pos": pos_mse.item(),
                    "loss neg": neg_mse.item(),
                    "loss reg": reg_mse.item(),
                    "loss cls": cls_mse.item()
                },
                step=current_iteration,
            )

            print(
                "\rIteration: {}, loss: {}".format(current_iteration, loss.item()),
                end="",
            )

            # generate visualization every N iterations
            if current_iteration % 250 == 0 and show_test_images:
                with torch.no_grad():
                    # test_images: torch.Size([5, 3, 480, 640])
                    # out: torch.Size([5, 20, 15, 20])
                    out = detector(test_images).cpu()
                    bbs = detector.decode_output(out, 0.5)
                    # attr of bbs: width, height, x, y, category

                    for i, test_image in enumerate(test_images):
                        figure, ax = plt.subplots(1)
                        plt.imshow(test_image.cpu().permute(1, 2, 0))
                        plt.imshow(
                            out[i, 4, :, :],
                            interpolation="nearest",
                            extent=(0, 640, 480, 0),
                            alpha=0.7,
                        )

                        # add bounding boxes
                        utils.add_bounding_boxes(ax, bbs[i], category_dict=CATEGORY_DICT)

                        wandb.log(
                            {"test_img_{i}".format(i=i): figure}, step=current_iteration
                        )
                        plt.close()

            current_iteration += 1
            if current_iteration > max_iterations:
                break

    print("\nTraining completed (max iterations reached)")

    model_path = "{}.pt".format(run_name)
    utils.save_model(detector, model_path)

    if device == "cpu":
        wandb.save(model_path)

    print("Model weights saved at {}".format(model_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    device = parser.add_mutually_exclusive_group(required=True)
    device.add_argument("--cpu", dest="device", action="store_const", const="cpu")
    device.add_argument("--gpu", dest="device", action="store_const", const="cuda")
    parser.add_argument("MAX_ITER", default=3000)
    args = parser.parse_args()

    train(int(args.MAX_ITER), args.device)
