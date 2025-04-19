"""
Leaksmy Heng
CS5330
4/13/2025
This is implementing ResNet50 from scratch

Reference:
https://www.kaggle.com/code/mishki/resnet-keras-code-from-scratch-train-on-gpu
https://www.youtube.com/watch?v=DkNIBBBvcPs&t=54s
"""
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from deep_learning_framework.constants import Constants
from deep_learning_framework.data_pytorch import ASLDataSet

EXPANSION = 4

class BottleNeckBlock(nn.Module):
    """A ResNet bottleneck block."""

    def __init__(self, in_channels: int, out_channels: int, identity_downsampling: Optional[nn.Module]=None, stride=1, activation: Callable = nn.ReLU):
        """
        Constructor for a ResNet bottleneck block.

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param identity_downsampling: optional downsampling layer for identity connection
        :param stride: stride for 3*3 convolution layer
        :param activation: activation function which is ReLu
        """
        super().__init__()
        # Layer configuration. First layer 1x1 Conv - Reduces dimensions
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # batch normalization
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        # Layer configuration. Second layer 3x3 Conv - Spatial processing
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        # Batch normalization
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        # Layer configuration. Third layer - Dimension restoration/expansion
        self.conv3 = nn.Conv2d(out_channels, out_channels * EXPANSION, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * EXPANSION)

        self.relu = activation()
        self.identity_downsampling = identity_downsampling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation."""
        # save the original input tensor (x) as identity
        identity = x

        # first convolutional layer input x and output out
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        # second convolutional layer
        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        # third convolutional layer
        out = self.conv3(out)
        out = self.batch_norm3(out)

        # in case the input and output has different shape, conduct downsampling operation
        if self.identity_downsampling is not None:
            identity = self.identity_downsampling(identity)

        # add input (identity) back to the output (ResNet does addition)
        out += identity
        out = self.relu(out)
        return out

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_channels={self.conv1.in_channels}, "
                f"out_channels={self.conv2.out_channels}, expansion={EXPANSION})")


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, number_of_classes):
        """
        Constructor of ResNet

        :param block: Bottleneck block
        :param layers: how many times we want to use the bock. For ResNet50: [3,4,6,3]
        :param image_channels: the channel of the image like RGB or something like that.
        :param number_of_classes: output size
        """
        super().__init__()
        self.in_channels = 64

        # initialize convolution and max pool
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # make layers ResNet-50 layers configuration: [3, 4, 6, 3]
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        self.average_pool = nn.AdaptiveMaxPool2d((1,1))
        self.fc = nn.Linear(512 * EXPANSION, number_of_classes)

    def _make_layer(self, block, number_of_residual_block, out_channels, stride):
        identity_downsampling = None
        layers = []

        # Down sampling for the first block in each layer (except the first layer)
        if stride != 1 or self.in_channels != out_channels * EXPANSION:
            identity_downsampling = nn.Sequential(nn.Conv2d(self.in_channels, out_channels * EXPANSION, kernel_size=1, stride=stride),
                                                  nn.BatchNorm2d(out_channels * EXPANSION))

        # First block in the layer
        layers.append(block(self.in_channels, out_channels, identity_downsampling, stride))
        # update in channel to be out channel * 4
        self.in_channels = out_channels * EXPANSION

        # append remaining block
        for _ in range(number_of_residual_block - 1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.average_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNet50(image_channels=3, number_of_classes=1000):
    return ResNet(BottleNeckBlock, [3, 4, 6, 3], image_channels, number_of_classes)


def train(epoch: int, device, network: ResNet50, train_loader, optimizer: torch.optim, train_losses: list, train_counter: list, is_save=True):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # move data to device .ie GPU if exists
        data, target = data.to(device), target.to(device)

        # resets the gradients of all optimized parameters to zero
        # this is because this is train in different epoch so each epoch should be started with 0
        optimizer.zero_grad()
        # Takes the input data and passes it through all the layers of the network
        # the output of our network
        output = network(data)
        # compute a cross  entropy loss
        loss = F.cross_entropy(output, target)
        # using backward propagation to get the loss and use the optimizer step
        loss.backward()
        optimizer.step()

        # log the information. Doing the mod to avoid way to noisy log
        if batch_idx % Constants.LOG_INTERVAL == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader)}%)]\tLoss: {loss.item()}')
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))

            # Task 1D - Save the network to a file
            # save the state dictionaries of both the neural network model and the optimizer
            if is_save:
                torch.save(network.state_dict(), 'model.pth')
                torch.save(optimizer.state_dict(), 'optimizer.pth')

    return train_losses, train_counter


def validate(network: ResNet50, val_loader, val_losses, device='cuda', get_accuracy=False):
    """Function to test the network.

    :param network: your deep network
    :param val_loader: test dataset
    """
    # set neural network model to evaluation mode as prior to this we set it to training mode
    network.eval()
    val_loss = 0
    correct = 0

    # disables gradient computation as we are not training it
    with torch.no_grad():
        for data, target in val_loader:
            # have this use GPU if found.
            data, target = data.to(device), target.to(device)
            output = network(data)
            # compute the negative log function loss
            val_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    accuracy = 100 * correct / len(val_loader.dataset)
    print(f'\nTest set: Avg. loss: {val_loss}, Accuracy: {correct}/{len(val_loader.dataset)} ({accuracy}%)\n')
    if get_accuracy:
        return val_losses, accuracy
    return val_loader, val_losses


def train_network():
    """
    Function to train the ResNet that we have written above.
    """
    # initialized resnet network
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    network = ResNet50().to(device)

    # initialize optimizer to use
    optimizer = torch.optim.Adam(network.parameters())

    # get train, validate and test data
    train_loader = ASLDataSet().get_training_dataset()
    validate_loader = ASLDataSet().get_validating_dataset()
    test_loader = ASLDataSet().get_testing_dataset()

    # keeping track of training losses and test losses
    train_losses = []
    train_counter = []
    val_losses = []
    val_counter = [i * len(train_loader.dataset) for i in range(Constants.EPOCHS + 1)]

    # one epoch at a time
    val_loader, val_losses = validate(network, validate_loader, val_losses)
    for epoch in range(1, Constants.EPOCHS + 1):
        # train the network
        train_losses, train_counter = train(epoch, device, network, train_loader, optimizer, train_losses, train_counter)
        # validate the network
        val_loader, val_losses = validate(network, validate_loader, val_losses)

    # Plot the training and testing accuracy in a graph
    print('Train counter is: ', train_counter)
    print('Train loss is:', train_losses)
    print('Validating counter is:', val_counter)
    print(val_losses)
    plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(val_counter, val_losses, color='red')
    plt.legend(['Train Loss', 'Validating Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('cross entropy loss')
    plt.show(block=True)


def test_network(network_path):
    """
    Function to test network. This should return a confusion metrics plot.

    :param network: network we train. Just load it from model.pth
    """
    use_cuda = torch.cuda.is_available()
    print("cuda" if use_cuda else "cpu")
    device = torch.device("cuda" if use_cuda else "cpu")
    network = ResNet50().to(device)

    # load the save model
    network.load_state_dict(torch.load('model.pth'))




def main():
    """Main function used to train the network."""
    print('Start training the data')
    # train_network()
    print('Finish training data')


if __name__ == "__main__":
    main()
