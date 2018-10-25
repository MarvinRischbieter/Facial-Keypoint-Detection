## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class OutputNet(nn.Module):
    def __init__(self, input_n, output_n, mid_n=None, dropout_p=0.5):
        super(OutputNet, self).__init__()
        """
        Several fc layers to make an output out of the encoder results
        Args:
            input_n (int): Number of elements in the input tensor
            output_n (int): Number of elements to output
            mid_n (int): Number of neurons in the middle fc layer.
                Use (input_n+output_n)/2 if mid_n is None
            dropout_p (float): Dropout probability
        """
        if mid_n is None:
            mid_n = int((input_n+output_n)/2)
        self.fc1 = nn.Linear(input_n, mid_n)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(mid_n, mid_n//2)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.fc3 = nn.Linear(mid_n//2, output_n)
    
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class Net(nn.Module):

    def __init__(self, input_shape=(224, 224), output_len=136,
                 dropout_p=0.5, n_blocks=5):
        super(Net, self).__init__()
        assert input_shape[0] == input_shape[1], "Input should be a square"
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.encoders = nn.ModuleList([])
        for i in range(n_blocks):
            block = self.make_encoder(32+i*32, 32+(i+1)*32)
            self.encoders.append(block)

        self.dropout = nn.Dropout(p=dropout_p)
        s = input_shape[0]//2**n_blocks
        self.output_layers = OutputNet((n_blocks+1)*32*s*s,
                                       output_len, 1024, dropout_p)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
    def make_encoder(self, input_layers, output_layers):
        conv = nn.Conv2d(input_layers, output_layers, kernel_size=3,
                         padding=1)
        bn = nn.BatchNorm2d(output_layers)
        relu = nn.ReLU(inplace=True)
        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        layers = [conv, bn, relu, maxpool]
        return nn.Sequential(*layers)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1(x)
        for enc in self.encoders:
            x = enc(x)
        x = self.dropout(x)
        x = self.output_layers(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
