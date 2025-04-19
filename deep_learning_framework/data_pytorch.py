"""
Leaksmy Heng
CS5330
4/17/2025
In data.py, I used Keras because I thought we would be able to train the data using the pretrain ResNet50.
However, upon training, I got really low accuracy rate, with further investigation, I found out that the pre-train
ResNet50 on imagenet dataset does not have the ASL data; therefore, the weight is really off.

I decided to create and train the ResNet50 from scratch written in model.py
As a result, I can't use Keras; therefore, I need a separate code to load the data using pytorch
"""


import torch
from matplotlib import pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import transforms

from deep_learning_framework.constants import Constants


# Conduct a series of transformation using transforms.Compose() including converting images to tensor,
# resize image to fit ResNet50 224x224, and normalize the image.
# I did not do horizontal flip or vertical flip for augmenting because this is ASL and different possition of hand is
# probably has different meaning.
# Use imagenet normalize for now [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = ImageFolder(root=r'C:\Users\hengl\OneDrive\Documents\GitHub\Real-Time-sign-language-translation\deep_learning_framework\data\training_data', transform=transform)
val_dataset = ImageFolder(root=r'C:\Users\hengl\OneDrive\Documents\GitHub\Real-Time-sign-language-translation\deep_learning_framework\data\validating_data', transform=transform)
test_dataset = ImageFolder(root=r'C:\Users\hengl\OneDrive\Documents\GitHub\Real-Time-sign-language-translation\deep_learning_framework\data\testing_data', transform=transform)


class ASLDataSet:
    """
    Function to get data ASL data from deep_learning_framework directory.
    """
    def __init__(self):
        self.batch_size_train = Constants.BATCH_SIZE
        self.batch_size_test = Constants.BATCH_SIZE_TEST
        self.random_seed = Constants.RANDOM_SEED
        torch.manual_seed(self.random_seed)

    def get_training_dataset(self, is_shuffle=True):
        """Function to get ASL train dataset. By default, this dataset is shuffled."""
        try:
            training_data = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size_train, shuffle=is_shuffle)

        except Exception as e:
            raise ValueError(f'Not able to get training data set with error: {e}')

        return training_data

    def get_validating_dataset(self, is_shuffle=True):
        """Function to get ASL validating dataset. By default, this dataset is shuffled."""
        try:
            validating_data = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size_train, shuffle=is_shuffle)

        except Exception as e:
            raise ValueError(f'Not able to get validating data set with error: {e}')

        return validating_data

    def get_testing_dataset(self, is_shuffle=False):
        """Function to get ASL test dataset. By default, this dataset is not shuffled."""
        try:
            testing_data = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.batch_size_train, shuffle=is_shuffle)

        except Exception as e:
            raise ValueError(f'Not able to get testing data set with error: {e}')

        return testing_data


    def plot(self):
        """Function to plot the data to see what it looks like. This is plotting 6 data in test dataset."""
        test_loader = self.get_testing_dataset(is_shuffle=False)
        examples = enumerate(test_loader)
        batch_idx, (example_data, example_targets) = next(examples)

        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
            plt.title("Label: {}".format(example_targets[i]))
            plt.xticks([])
            plt.yticks([])

        plt.tight_layout()
        plt.show(block=True)


# ASLDataSet().plot()
