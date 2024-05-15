import os
from PIL import Image
from torch.utils.data import Dataset

import random


class FireRiskDataset(Dataset):
    def __init__(self, root_dir, transform=None, sample_size=None, resample=False, label_dict=[]):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            sample_size (int, optional): Number of samples to include (randomly selected).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.label_dict = label_dict

        # Read all images and labels
        for label in os.listdir(root_dir):
            class_path = os.path.join(root_dir, label)
            if os.path.isdir(class_path):
                for img in os.listdir(class_path):
                    self.images.append(os.path.join(class_path, img))
                    self.labels.append(self.label_dict[label])

        # Compute class counts
        self.class_counts = [0] * len(self.label_dict)
        for label in self.labels:
            self.class_counts[label] += 1
            
        # If sample size is specified, randomly select that many samples
        if sample_size is not None and sample_size < len(self.images):
            sample_indices = random.sample(range(len(self.images)), sample_size)
            self.images = [self.images[i] for i in sample_indices]
            self.labels = [self.labels[i] for i in sample_indices]
        
        # Resample the dataset to have equal representation of each class     
        if resample:
            self.resample()

    # Get the number of samples
    def __len__(self):
        return len(self.images)

    # Load image and label
    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(img_name)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
    # Resample the dataset to have equal representation of each class
    def resample(self):
        # Find the class with the fewest samples
        min_class_count = min(self.class_counts)
        
        # Create a new list of images and labels
        new_images = []
        new_labels = []
        
        for i in range(len(self.label_dict)):
            # Find all indices of the current class
            indices = [idx for idx, label in enumerate(self.labels) if label == i]
            # Randomly sample min_class_count indices or the total number of samples in the class, whichever is smaller
            sample_size = min(min_class_count, len(indices))
            indices = random.sample(indices, sample_size)
            # Add the sampled images and labels to the new list
            new_images.extend([self.images[idx] for idx in indices])
            new_labels.extend([self.labels[idx] for idx in indices])
            
        # Update the dataset with the new images and labels
        self.images = new_images
        self.labels = new_labels
        self.class_counts = [len([label for label in new_labels if label == i]) for i in range(len(self.label_dict))]
        
        return
    