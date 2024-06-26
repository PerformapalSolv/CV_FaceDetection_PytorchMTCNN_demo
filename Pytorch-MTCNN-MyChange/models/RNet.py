import torch.nn as nn
import torch


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=28, kernel_size=(3, 3))
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(in_channels=28, out_channels=48, kernel_size=(3, 3))
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(2, 2))
        self.prelu3 = nn.PReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=576, out_features=128)
        self.class_fc = nn.Linear(in_features=128, out_features=2)
        self.bbox_fc = nn.Linear(in_features=128, out_features=4)
        # self.landmark_fc = nn.Linear(in_features=128, out_features=10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu2(self.conv2(x))
        x = self.pool2(x)
        x = self.prelu3(self.conv3(x))
        x = self.flatten(x)
        x = self.fc(x)
        # 分类是否人脸的卷积输出层
        class_out = self.class_fc(x)
        # 人脸box的回归卷积输出层
        bbox_out = self.bbox_fc(x)
        # 5个关键点的回归卷积输出层
        # landmark_out = self.landmark_fc(x)
        return class_out, bbox_out
        # landmark_out
