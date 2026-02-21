import torch

'''
Change the data folder location and training variables here

Source for CNN:
https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
https://www.geeksforgeeks.org/deep-learning/understanding-the-forward-function-output-in-pytorch/
'''

DATA_FOLDER = "./data/"
CLIPS_FOLDER = "./clips/"
TAKEDOWN_CLASSIFIER = 1
NORMAL_CLASSIFIER = 0
TRAINING_DATA_PERCENT = 0.8
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 5

class BurnoutCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2) #cut height and width in half
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 53 * 53, 64) #convolution layer output formula: https://stackoverflow.com/questions/53580088/calculate-the-output-size-in-convolution-layer, don't forget maxpool halves it twice
        self.fc2 = torch.nn.Linear(64, 2) #have to use 2 classes here, for takedown or normal

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x