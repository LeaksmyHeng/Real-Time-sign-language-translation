"""
Leaksmy Heng
CS5330
4/13/2025
Model used to train the dataset in kagglehub folder
"""
from glob import glob

import keras
from keras.src.applications.resnet import ResNet50
import pandas as pd
from keras.src.callbacks import EarlyStopping
from keras.src.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.python.keras.saving.save import load_model

from deep_learning_framework.constants import Constants
from deep_learning_framework.data import Dataset


def get_number_of_classes(path: str = r'..\deep_learning_framework\data\validating_data\*') -> int:
    """
    Function to get the number of classes to do the classification.

    :param path: directory to get the number of classes. Default to the validating data.

    :return: number of classes
    """
    classes = glob(path)
    number_of_classes = len(classes)
    print(f'Total number of classes to classify: {number_of_classes}')
    return number_of_classes


def ResNet_implementation(train_generator, validation_generator):
    """
    Function to implement ResNet. Since ResNet50 is already pretrained and could be found in keras.src.applications.resnet,
    we will use that.

    :return:

    # https://vijayabhaskar96.medium.com/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720
    https://www.youtube.com/watch?v=5SJAPmQy7xs

    """
    ## Load the pre-trained ResNet50 model, exclude the top layers
    base_model = ResNet50(weights='imagenet',
                          include_top=False,
                          input_shape=(224, 224, 3))
    # print(base_model.summary())

    # freeze the base model layers
    base_model.trainable = False

    ## build the model
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    prediction_layers = keras.layers.Dense(get_number_of_classes(), activation='softmax')(x)

    model = keras.models.Model(inputs=base_model.input, outputs=prediction_layers)

    # Compile our model
    model.compile(
        optimizer=Adam(learning_rate=Constants.LEARNING_RATE),  # Lower LR for fine-tuning
        loss='categorical_crossentropy',                        # For multi-class classification
        metrics=['accuracy']
    )

    # train the model
    resnet_history = model.fit(train_generator,
                               steps_per_epoch=train_generator.samples // Constants.BATCH_SIZE,
                               validation_data=validation_generator,
                               validation_steps=validation_generator.samples // Constants.BATCH_SIZE,
                               epochs=Constants.EPOCHS,
                               callbacks=[EarlyStopping(monitor='val_loss', patience=4, verbose=1)],
                               verbose=1
                               )

    # save the model
    model.save('resnet50.h5')

    metrics = pd.DataFrame(model.history.history)
    print("The model ResNet50 metrics are")
    print(metrics)

    # Plotting training and validation accuracy
    plt.plot(metrics['accuracy'], label='Train Accuracy')
    plt.plot(metrics['val_accuracy'], label='Validation Accuracy')
    plt.title('ResNet50 Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plotting training and validation loss
    plt.plot(metrics['loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Validation Loss')
    plt.title('ResNet50 Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def test(test_generator, model_to_be_loaded):
    """
    Function to tests the model

    :param test_generator: testing data
    :param model_to_be_loaded: the model you have trained and saved.
    """
    # load the save model
    model = load_model(model_to_be_loaded)
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_acc:.2%}")
    predictions = model.predict(test_generator)

    cout = 0
    for images, labels in test_generator:
        preds = model.predict(images)
        plt.imshow(images[0].astype('uint8'))  # Display first image
        plt.title(f"True: {labels[0]}, Pred: {preds[0]}")
        plt.show()
        cout += 1
        if cout >= 5:
            break



data_path = r'C:\Users\hengl\OneDrive\Documents\GitHub\Real-Time-sign-language-translation\deep_learning_framework\data\asl_alphabet'
output_path = r'C:\Users\hengl\OneDrive\Documents\GitHub\Real-Time-sign-language-translation\deep_learning_framework\data'
train_generator, validating_generator, testing_generator = Dataset(dataset_folder=data_path).data_preprocessing(
    training_directory=r'..\deep_learning_framework\data\training_data',
    testing_directory=r'..\deep_learning_framework\data\testing_data',
    validating_directory=r'..\deep_learning_framework\data\validating_data'
)

ResNet_implementation(train_generator, validating_generator)
# test(testing_generator, 'resnet50.h5')