import torch.nn as nn
import torch


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.maxpool_layer1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.maxpool_layer2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc_layer1 = nn.Linear(1024, 128)
        self.relu = nn.ReLU()
        self.fc_layer2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.maxpool_layer1(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (3, 3), bias=True, padding=1)
        torch.nn.init.xavier_uniform(self.conv1.weight)
        self.conv2 = nn.Conv2d(64, 128, (3, 3), bias=True, padding=1)
        torch.nn.init.xavier_uniform(self.conv2.weight)
        self.conv3 = nn.Conv2d(128, 256, (3, 3), bias=True, padding=1)
        torch.nn.init.xavier_uniform(self.conv3.weight)
        self.conv4 = nn.Conv2d(256, 512, (3, 3), bias=True, padding=1)
        torch.nn.init.xavier_uniform(self.conv4.weight)
        self.fc1 = nn.Linear(2097152, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.dropout(x)
        x = self.activation(self.conv2(x))
        x = self.dropout(x)
        x = self.activation(self.conv3(x))
        x = self.dropout(x)
        x = self.activation(self.conv4(x))
        x = self.dropout(x)
        x = x.reshape(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        return x


class SimpleNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (3, 3), bias=True, padding=1)
        self.fc1 = nn.Linear(262144, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.dropout(x)
        x = x.reshape(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        return x


class MicroVGG(nn.Module):
    """
    VGG like model, inspired from https://poloclub.github.io/cnn-explainer/
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),

            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 16 * 16,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
