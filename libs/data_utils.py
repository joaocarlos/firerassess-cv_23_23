import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from libs.dataset import FireRiskDataset


# Function to display an image
def imshow(img):
    # These values should match the normalization parameters used during preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Reverse the normalization process
    img = img.numpy()
    img = (img * std[:, None, None]) + mean[:, None, None]  # Apply std first, then mean

    # Convert tensor image to numpy and change from (C, H, W) to (H, W, C) for displaying
    img = np.transpose(img, (1, 2, 0))  # Rearrange the dimensions
    img = np.clip(img, 0, 1)  # Clip values to be in the range [0, 1]

    # Display the image
    plt.imshow(img)
    plt.show()


# Function to save the model
def saveModel(model: nn.Module, path):
    torch.save(model.state_dict(), path)


# Save model information to a file
def save_model_info(model, optimizer, epoch, args, path):
    model_info = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": {
            "batch_size": args["batch"],
            "lr": args["lr"],
            "weight_decay": args["weight_decay"],
            "smoothing_factor": args["smoothing_factor"],
        },
    }
    torch.save(model_info, path)


# Save metrics to a file
def save_metrics(metrics, path):
    with open(path, "w") as f:
        for metric in metrics:
            f.write(f"{metric}\n")


# Load metrics from a file
def load_metrics(path):
    with open(path, "r") as f:
        metrics = f.readlines()
    metrics = [float(metric.strip()) for metric in metrics]
    return metrics


# Compute the Normalization parameters
def compute_normalize(label_dict, root_dir: str):
    dataset = FireRiskDataset(
        root_dir=root_dir,
        label_dict=label_dict,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=2)

    mean = 0.0
    std = 0.0
    nb_samples = 0

    for data in loader:
        data = data[0]
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std
