import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import libs.data_utils as du
from libs.pytorchtools import EarlyStopping


def train(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    start_epoch: int,
    num_epochs: int,
    path_model: str,
    verbatim: bool = True,
):
    # Set the model to training mode
    model.train()
    # Move the model to the device
    model.to(device)

    # Initialize the best accuracy and loss
    best_accuracy = 0.0
    best_loss = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        train_acc = 0.0
        train_loss = 0.0
        for ibatch, (images, labels) in enumerate(dataloader, 0):

            # get the inputs
            images = images.to(device)
            labels = labels.to(device)

            # 1. Forward pass: compute predicted outputs by passing inputs to the model
            y_pred = model(images)

            # 2. Calculate  and accumulate loss
            loss = loss_fn(y_pred, labels)
            train_loss += loss.item()

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

            # Calculate and accumulate accuracy metric across all batches
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == labels).sum().item() / len(y_pred)

        # Adjust metrics to get average loss and accuracy per batch
        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)

        # If we want to save the model at each epoch
        # path = "./models/FireRisk_InceptionResnetV2_" + str(epoch) + ".pth"
        # du.saveModel(model, path=path)

        # If we want to save the model if the accuracy is the best
        if train_acc > best_accuracy:
            path = str(path_model) + "Best.pth"
            best_loss = train_loss
            best_accuracy = train_acc
            best_epoch = epoch + start_epoch
            du.saveModel(model, path=path)
            if verbatim:
                print(
                    "Best Epoch #",
                    best_epoch,
                    " Loss=",
                    best_loss,
                    " Accu=",
                    best_accuracy,
                )

    return best_loss, best_accuracy, best_epoch


# Function to test the model with the test dataset and print the accuracy for the test images
def test(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    verbatim=True,
):

    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    pred_labels = []
    with torch.no_grad():
        for data in dataloader:

            # get the inputs
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            # 1. run the model on the test set to predict labels
            y_pred = model(images)

            # 2. Calculate and accumulate loss
            loss = loss_fn(y_pred, labels)
            test_loss += loss.item()

            # 3. Calculate and accumulate accuracy
            y_pred_class = torch.argmax(y_pred, dim=1)
            test_acc += (y_pred_class == labels).sum().item()
            pred_labels = y_pred_class.tolist()

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    if verbatim:
        print("Loss =", test_loss, "  Accuracy=", test_acc)
    return pred_labels, test_loss, test_acc


def evaluate_model(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn: torch.nn.Module,
):
    """
    Evaluate a trained model on a validation set.

    Parameters:
    model (torch.nn.Module): The trained model that needs to be evaluated.
    val_loader (torch.utils.data.DataLoader): The DataLoader object that loads the validation data.
    device (torch.device): The device (CPU or GPU) where the model and data are loaded.

    Returns:
    list: A list of all true labels from the validation set.
    list: A list of all predictions made by the model on the validation set.
    """

    # Set model to evaluate mode
    model.eval()

    # Initialize counters for correct predictions and total number of samples
    correct = 0
    total = 0

    # Initialize lists to store all labels and predictions
    all_labels = []
    all_preds = []

    # Disable gradient computation since we are in evaluation mode
    with torch.no_grad():
        # Iterate over batches of data in the validation loader
        for inputs, labels in val_loader:
            # Move inputs and labels to the specified device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Make predictions using the model
            outputs = model(inputs)

            # Get the predicted class with the highest score
            predicted = torch.argmax(outputs, dim=1)

            # Update total number of samples and number of correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Add true labels and predictions to their respective lists
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(predicted.cpu().numpy().tolist())

    # Calculate accuracy as percentage of correct predictions
    accuracy = 100 * correct / total

    # Print accuracy
    print(f"Accuracy on validation set: {accuracy}%")

    # Return lists of true labels and predictions
    return all_labels, all_preds


def train_and_eval(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    start_epoch: int,
    num_epochs: int,
    path_model: str,
    early_stop: bool = False,
    patience: int = 20,
    verbatim: bool = True,
    model_args: dict = {},
):
    # Set the model to training mode
    model.train()
    # Move the model to the device
    model.to(device)

    if early_stop:
        # initialize the early_stopping object
        early_stopping = EarlyStopping(
            patience=patience, verbose=verbatim, path=str(path_model) + "checkpoint.pth"
        )

    # Initialize the best metrics
    best_accuracy = 0.0
    best_loss = np.Inf
    best_epoch = 0

    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []

    for epoch in range(num_epochs):
        train_acc = 0.0
        train_loss = 0.0

        # Training phase
        model.train()
        for ibatch, (images, labels) in enumerate(train_loader):
            # Move inputs and labels to the device
            images, labels = images.to(device), labels.to(device)

            # 1. Forward pass
            y_pred = model(images)

            # 2. Loss calculation and accumulation
            loss = loss_fn(y_pred, labels)
            train_loss += loss.item()

            # 3. Backpropagation and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 4. Calculate training accuracy
            y_pred_class = torch.argmax(y_pred, dim=1)
            train_acc += (y_pred_class == labels).sum().item() / len(labels)

        # Average training loss and accuracy per batch
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # Validation phase
        val_acc = 0.0
        val_loss = 0.0

        model.eval()  # Set to evaluation mode
        with torch.no_grad():  # Disable gradients
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                y_pred = model(images)

                # Accumulate validation loss
                loss = loss_fn(y_pred, labels)
                val_loss += loss.item()

                # Calculate validation accuracy
                y_pred_class = torch.argmax(y_pred, dim=1)
                val_acc += (y_pred_class == labels).sum().item() / len(labels)

            val_loss /= len(val_loader)
            val_acc /= len(val_loader)

        # store the validation accuracy for future plot
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

        # Display training and validation metrics for monitoring
        if verbatim:
            print(
                f"Epoch [{epoch + start_epoch + 1} / {start_epoch + num_epochs}]: Train Loss={train_loss}, Train Accu={train_acc}, Validation Loss={val_loss}, Validation Accu={val_acc}"
            )

        if val_loss < best_loss:
            best_loss = val_loss
            best_accuracy = val_acc
            best_epoch = epoch + start_epoch

        if early_stop:
            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping...")
                break
        else:
            # Check if validation accuracy is the best so far
            if val_loss < best_loss:
                path = str(path_model) + str(best_epoch + 1) + ".pth"
                du.saveModel(model, path=path)
                if verbatim:
                    print(
                        "Best Epoch #",
                        best_epoch + 1,
                        " Loss=",
                        best_loss,
                        " Accu=",
                        best_accuracy,
                    )
                    print("Saving model...")

    # Save the accuracy and loss lists to a csv using pandas
    du.save_metrics(train_acc_list, str(path_model) + "train_acc.csv")
    du.save_metrics(train_loss_list, str(path_model) + "train_loss.csv")
    du.save_metrics(val_acc_list, str(path_model) + "val_acc.csv")
    du.save_metrics(val_loss_list, str(path_model) + "val_loss.csv")

    return (
        best_loss,
        best_accuracy,
        best_epoch,
        train_acc_list,
        train_loss_list,
        val_acc_list,
        val_loss_list,
    )


def predict_image_class(class_name, dataset, model, device, transform, classes):
    """
    Select a random image from a specified class in the validation dataset, perform inference, and display the image along with predictions.

    Parameters:
    - class_name (str): The name of the class to select the image from.
    - dataset (Dataset): The validation dataset which includes 'labels' and 'images'.
    - model (torch.nn.Module): The trained model to use for prediction.
    - device (torch.device): The device to perform computation on.
    - transform (torchvision.transforms): The transformations to apply to the image for model input.

    Returns:
    - None
    """

    # Define classes
    # classes = [
    #     "Very_High",
    #     "High",
    #     "Moderate",
    #     "Low",
    #     "Very_Low",
    #     "Non-burnable",
    #     "Water",
    # ]

    # Find the index of the class
    class_index = classes.index(class_name)

    # Get all indices of images in the specified class
    class_indices = [
        i for i, label in enumerate(dataset.labels) if label == class_index
    ]

    # Select a random index from these class indices
    image_index = random.choice(class_indices)
    image_path = dataset.images[image_index]

    # Load the image
    image = Image.open(image_path)

    # Transform the image for the model input
    image_transformed = transform(image).unsqueeze(0).to(device)

    # Perform model inference
    model = model.to(device)
    model.eval()
    output = model(image_transformed)

    # Get the predicted class
    _, predicted = torch.max(output, 1)
    predicted_class = classes[predicted.item()]

    # Get file name
    file_name = os.path.basename(image_path)

    # Print the outputs
    print("Predicted class:", predicted_class)
    print("File name:", file_name)

    # Display the image
    du.imshow(torchvision.utils.make_grid(image_transformed.cpu().detach()))


def plot_confusion_matrix(
    true_labels: list, pred_labels: list, classes: list, best_epoch: int
):
    """
    Plots a confusion matrix using true and predicted labels.

    Parameters:
    - true_labels (list): List of true labels.
    - pred_labels (list): List of predicted labels.
    - classes (list): List of class names as strings.
    """
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    # Add a title with the best epoch
    plt.title(f"Confusion Matrix (Best Epoch: {best_epoch})")
    plt.xticks(rotation=45)
    plt.show()
