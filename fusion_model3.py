
import torch
import torch.nn as nn
import torch.nn.functional as F
class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()


        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, f1_low, f1_high):
        f_ = torch.cat((f1_low, f1_high), dim=1)
        f=self.maxpool(f_)
        fusion1 = self.conv_layer1(f)
        fusion2=self.conv_block1(fusion1)
        fusion2=self.upsample(fusion2)
        f_fusion = self.conv_block2(fusion2)
        f =f_fusion+f_

        return f