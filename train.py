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
from config import *


def train(max_iter, device="cpu"):
    """Train the network.

    Args:
        max_iter: The maximum of training iterations.
        device: The device to train on."""

    wandb.init(project="detector_baseline")

    # Init model
    detector = Detector(NUM_CATEGORIES).to(device)

    wandb.watch(detector)

    dataset = CocoDetection(
        root="./dd2419_coco/training",
        annFile="./dd2419_coco/annotations/training.json",
        transforms=detector.input_transform,
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # training params
    max_iterations = wandb.config.max_iterations = max_iter

    learning_rate = wandb.config.learning_rate = 1e-4
    weight_reg = wandb.config.weight_reg = 1
    weight_noobj = wandb.config.weight_noobj = 1
    weight_cls = wandb.config.weight_cls = 1

    # run name (to easily identify model later)
    time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    run_name = wandb.config.run_name = "saved_models/det_{}".format(time_string)

    # init optimizer
    optimizer = torch.optim.Adam(detector.parameters(), lr=learning_rate)

    # load training images
    train_images = []
    show_training_images = True
    directory = "./dd2419_coco/training"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # get 1 image from 100
    for idx, file_name in enumerate(os.listdir(directory)[::100]):
        if file_name.endswith(".jpg"):
            file_path = os.path.join(directory, file_name)
            train_image = Image.open(file_path)
            train_images.append(TF.to_tensor(train_image))
        # if idx >= 9:    # only use 5 images
        #     break

    if train_images:
        train_images = torch.stack(train_images)
        train_images = train_images.to(device)
        show_train_images = True

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
        detector.train()

        train_cls_all = 0
        train_cls_correct = 0

        for img_batch, target_batch in dataloader:
            img_batch = img_batch.to(device)  # torch.Size([8, 3, 480, 640])
            target_batch = target_batch.to(device)  # Batch, 5+category, 15, 20

            # run network
            out = detector(img_batch)  # torch.Size([8, 5+category, 15, 20])

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
                out[
                    pos_indices[0], 0:4, pos_indices[1], pos_indices[2]
                ],  # torch.Size([8, 4])
                # each [4]: out[xx[0][0], 0:4, xx[1][0], xx[2][0]
                target_batch[
                    pos_indices[0], 0:4, pos_indices[1], pos_indices[2]
                ],  # torch.Size([8, 4])
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
            out_cls = out[pos_indices[0], 5:, pos_indices[1], pos_indices[2]]
            label_cls = target_batch[pos_indices[0], 5:, pos_indices[1], pos_indices[2]]
            cls_mse = nn.functional.mse_loss(out_cls, label_cls)
            # class acc
            train_cls_predict = torch.argmax(out_cls, 1)
            train_cls_label = torch.argmax(label_cls, 1)
            train_cls_correct += torch.sum(train_cls_predict == train_cls_label).item()
            train_cls_all += train_cls_label.numel()

            loss = (
                pos_mse
                + weight_reg * reg_mse
                + weight_noobj * neg_mse
                + weight_cls * cls_mse
            )

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
                    "loss cls": cls_mse.item(),
                },
                step=current_iteration,
            )

            print(
                "\rIteration: {}, loss: {}".format(current_iteration, loss.item()),
                end="",
            )

            # generate visualization every N iterations
            show_images = show_train_images and show_test_images

            if current_iteration % 250 == 0 and show_images:
                with torch.no_grad():
                    detector.eval()
                    # test_images: torch.Size([5, 3, 480, 640])
                    # out: torch.Size([5, 20, 15, 20])
                    out = detector(train_images).cpu()  # training
                    bbs = detector.decode_output(out, CONF_THRESHOLD)
                    out_test = detector(test_images).cpu()  # test
                    bbs_test = detector.decode_output(out_test, CONF_THRESHOLD)
                    # attr of bbs: width, height, x, y, category

                    for i, image in enumerate(train_images):
                        figure, ax = plt.subplots(1)
                        plt.imshow(image.cpu().permute(1, 2, 0))
                        plt.imshow(
                            out[i, 4, :, :],
                            interpolation="nearest",
                            extent=(0, 640, 480, 0),
                            alpha=0.7,
                        )

                        # add bounding boxes
                        utils.add_bounding_boxes(
                            ax, bbs[i], category_dict=CATEGORY_DICT
                        )

                        wandb.log(
                            {"train_img_{i}".format(i=i): figure}, step=current_iteration
                        )
                        plt.close()

                    for i, image in enumerate(test_images):
                        figure, ax = plt.subplots(1)
                        plt.imshow(image.cpu().permute(1, 2, 0))
                        plt.imshow(
                            out_test[i, 4, :, :],
                            interpolation="nearest",
                            extent=(0, 640, 480, 0),
                            alpha=0.7,
                        )

                        # add bounding boxes
                        utils.add_bounding_boxes(
                            ax, bbs_test[i], category_dict=CATEGORY_DICT
                        )

                        wandb.log(
                            {"test_img_{i}".format(i=i): figure}, step=current_iteration
                        )
                        plt.close()
                    detector.train()

            current_iteration += 1
            if current_iteration > max_iterations:
                break

        training_acc = train_cls_correct/train_cls_all
        # TODO
        wandb.log(
            {
                "training acc": training_acc
            },
            step=current_iteration,
        )
        # if current_iteration % 200 == 0:
        #     print("training_acc: %f" % training_acc)

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
    parser.add_argument("MAX_ITER", type=int, default=3000)
    args = parser.parse_args()

    train(int(args.MAX_ITER), args.device)
