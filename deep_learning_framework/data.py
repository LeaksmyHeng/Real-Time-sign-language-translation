"""
Leaksmy Heng
CS5330
4/10/2025
function to get the data from kaggle and do data processing prior to train it
"""

import logging
import os
import shutil

import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.applications.resnet50 import preprocess_input
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

from deep_learning_framework.constants import Constants


logger = logging.getLogger(__name__)

# # Download latest version
# path = kagglehub.dataset_download("grassknoted/asl-alphabet")
# print("Path to dataset files:", path)


def splitting_dataset(output_path):
    """
    Since when we downloaded the dataset from kaggle, we do not have the folders for training, validating and testing
    so this function is used to create the folders

    :param output_path: path that we store the spit dataset.
    """
    try:
        # create directory to store test, validating and training dataset
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        for split in ("training", "validating", "testing"):
            child_path = output_path + fr'\{split}_data'
            if not os.path.exists(child_path):
                os.mkdir(child_path)

    except Exception as e:
        raise e


class Dataset:
    def __init__(self, dataset_folder,  training_percentage: int = 0.8, validation_percentage: int = 0.1, test_percentage: int = 0.1):
        """
        A constructor for dataset object. Dataset object is used to determine how many percent we want to split our data
        for training, validating and testing. By default, we split it in 0.7, 0.2 and 0.1 for train, validate and test
        respectively.

        :param dataset_folder: location of the dataset after downloading from kaggle
        :param training_percentage: percentage of data used to train the model
        :param validation_percentage: percentage of data used to validate the model
        :param test_percentage: percentage of data used to test the model
        """
        self.dataset_folder = dataset_folder
        self.training_percentage = training_percentage
        self.validating_percentage = validation_percentage
        self.testing_percentage = test_percentage

    def data_summarization(self):
        """
        Function to conduct summarization.
        Check if there is imbalances in data.
        """
        data_imbalance = {}
        counter = 0
        for folder in os.listdir(self.dataset_folder):
            label = os.path.join(self.dataset_folder, folder)
            label_counter = 0
            for alp in os.listdir(label):
                counter += 1
                label_counter += 1
            data_imbalance.setdefault(folder, label_counter)

        # Found out that there is no data imbalance here.
        # so I just proceed with regular splitting of data (.8 training, .1validating, .1testing)
        logger.info(data_imbalance)
        logger.info(f'Total number of data: {counter}')

    def data_splitting(self, output_path: str, skipping_splitting: bool = True):
        """
        This function is used to split data into training, validating and testing.
        By default, resNet50 and ResNet18 require input image of 224 * 224 pixel.

        :param output_path: output directly you want the data to be located.
        :param skipping_splitting: skip all the splitting in case we already did that once. By default, this is True.

        Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        https://www.w3schools.com/Python/matplotlib_subplot.asp
        """
        try:
            logger.info('Starting to do data splitting')
            # call splitting_dataset(output_path) to create train, validate and test folder
            splitting_dataset(output_path=output_path)

            # in each of the folder (train, validate and test), create each label
            if not skipping_splitting:
                for label in os.listdir(self.dataset_folder):
                    label_path = os.path.join(self.dataset_folder, label)

                    # Get all images from the current label
                    # Join the label_path with the image name .i.e C:\\Users\\data\\asl_alphabet\\Z\\Z999.jpg
                    images = [os.path.join(label_path, img) for img in os.listdir(label_path)]

                    # split data into train
                    train_images, temp_images = train_test_split(images, test_size=self.testing_percentage, random_state=42)
                    # split tem_images into validation and test images
                    validation_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

                    # Copy the data from the directory
                    for img_set, split_name in zip([train_images, validation_images, test_images], [fr'{output_path}\training_data', fr'{output_path}\validating_data', fr'{output_path}\testing_data']):
                        split_label_dir = os.path.join(split_name, label)
                        os.makedirs(split_label_dir, exist_ok=True)
                        for img in img_set:
                            shutil.copy(img, split_label_dir)

                    # delete the data from parent file after done to save some space
                    shutil.rmtree(label_path)

            # check the number of file in each of the data split (validating, training and testing) to check for
            # data imbalance
            data_count = {}
            for folder in os.listdir(output_path):
                if 'data' in folder:
                    if folder not in data_count:
                        data_count[folder] = {}

                    for alphabet in os.listdir(fr'{output_path}\{folder}'):
                        if alphabet not in data_count[folder]:
                            data_count[folder][alphabet] = 0

                        for img in os.listdir(fr'{output_path}\{folder}\{alphabet}'):
                            data_count[folder][alphabet] += 1
                            # I also want to see the type of image like color and shape
                            # so only show like 3 images each from different folder
                            image = cv2.imread(fr'{output_path}\{folder}\{alphabet}\{img}')
                            image_shape = image.shape
                            logger.info(f'Image shape {image_shape} for {img}')

            # plotting training data
            total_dtype = len(data_count)
            plt.figure(figsize=(6 * total_dtype, 8))
            for counter, (dtype, dtype_val) in enumerate(data_count.items(), 1):
                keys = dtype_val.keys()
                values = dtype_val.values()

                # set subplot before plotting
                plt.subplot(1, total_dtype, counter)
                plt.barh(keys, values, color='skyblue')
                plt.xlabel('Labels')
                plt.ylabel('Counts')
                plt.title(f'Bar chat of {dtype}')

            plt.tight_layout()
            plt.show(block=True)

            logger.info('Finishing doing data splitting')

        except Exception as e:
            raise e

    def data_preprocessing(self, training_directory, testing_directory, validating_directory, image_width: int=224, image_height: int=224, is_applying_augmenting: bool = False):
        """This part is to conduct data preprocessing so that we could use it seamlessly when training in ResNet.

        :param training_directory: the training directory of your images
        :param validating_directory: the validating_directory of your images
        :param testing_directory: the testing_directory of your image
        :param image_width: the width of the image. Since we are going to use ResNet, the w and h is 224 by default
        :param image_height: the height of the image
        :param is_applying_augmenting: this is to add data augmenting to our dataset

        https://vijayabhaskar96.medium.com/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720
        """
        # Does not need to do normalization of 1/255 as we used ResNet specific preprocessing
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        # conducting augmenting to training data set if need be
        if is_applying_augmenting:
            train_datagen = ImageDataGenerator(
                preprocessing_function=preprocess_input,
                rescale=1/255,
                rotation_range=20,      # roting random -20 to +20 degree to simulate different orientation
                width_shift_range=0.2,  # simulates vertical camera movement -20 to +20
                height_shift_range=0.2, # simulates horizontal camera movement -20 to +20
                shear_range=0.2,        # shear transformation
                zoom_range=0.2,         # random zoom at 80 - 120 scale
                horizontal_flip=True,   # random flip horizontally
                fill_mode='nearest'     # fill missing pixels after transforms
            )

        train_generator = train_datagen.flow_from_directory(
            directory=training_directory,
            target_size=(image_width, image_height),
            color_mode="rgb",
            batch_size=32,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )

        datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        validating_generator = datagen.flow_from_directory(
            directory=validating_directory,
            target_size=(image_width, image_height),
            color_mode="rgb",
            batch_size=32,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        testing_generator = datagen.flow_from_directory(
            directory=testing_directory,
            target_size=(image_width, image_height),
            color_mode="rgb",
            batch_size=Constants.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False,
            seed=Constants.SEED
        )

        print(f"Classes detected: {train_generator.class_indices}")
        print(f"Total training samples: {train_generator.samples}")

        return train_generator, validating_generator, testing_generator
