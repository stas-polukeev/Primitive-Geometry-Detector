import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapesDetector(torch.nn.Module):

    def __init__(self) -> None:
        super(ShapesDetector, self).__init__()
        self.relu = nn.LeakyReLU(0.01)
        self.pool = nn.MaxPool2d(2, 2)

        #256*3
        self.conv1 = nn.Sequential(*[
            nn.Conv2d(3, 64, 7, 2, 3)
            ,self.relu
            ,self.pool
        ])
        #64*64
        self.conv2_1= nn.Sequential(*[
            nn.Conv2d(64, 32, 1, 1, 0)
            ,self.relu
        ])
        self.conv2_3 = nn.Sequential(*[
            nn.Conv2d(64, 64, 3, 1, 1)
            ,self.relu
        ])
        
        self.conv2_5 = nn.Sequential(*[
            nn.Conv2d(64, 32, 5, 1, 2)
            ,self.relu
        ])
        #32 * 128
        self.conv3 = nn.Sequential(
                    nn.Conv2d(128, 128, 1, 1, 0)
                    ,self.relu
        )
        #32 * 128
        self.conv4_3 = nn.Sequential(*[
            nn.Conv2d(128, 96, 3, 1, 1)
            ,self.relu
            ,self.pool
        ])
        self.conv4_1 = nn.Sequential(*[
            nn.Conv2d(128, 32, 1, 1, 0)
            ,self.relu
            ,self.pool
        ])
        #16*128
        self.conv5 = nn.Sequential(
                    nn.Conv2d(128, 128, 1, 1, 0)
                    ,self.relu)
        #16*64

        self.conv6  = nn.Sequential(*[
            nn.Conv2d(128, 256, 3, 1, 1)
            ,self.relu
            ,self.pool
        ])
        #8*128
        self.conv7 = nn.Sequential(
                    nn.Conv2d(256, 128, 1, 1, 0)
                    ,self.relu)
        #8*64
        self.conv8 = nn.Conv2d(128, 9, 1, 1, 0)
        #8*9
    def forward(self, x):
        # x: (B, 3, 256, 256)
        x = self.conv1(x)
        #(B, 64, 64, 64)

        x1 = self.conv2_5(x)
        x2 = self.conv2_3(x)
        x3 = self.conv2_1(x)
        x = torch.concat((x1, x2, x3), dim=1)
        x = self.pool(x)

        x = self.conv3(x)

        x1 = self.conv4_3(x)
        x2 = self.conv4_1(x)
        x = torch.concat((x1, x2), dim=1)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)

        #(B, 9, 8, 8)
        #here the channels (0 - 4) a correspond to bounding box and confidence
        #probability and its parameters (x, y, w, h) in [0, 1]
        #channels (5-9) correspond to 4 classes probabilities
        confidence = torch.sigmoid(x[:, 0, :, :]).unsqueeze(1) #object existanse confidence
        bbox = torch.sigmoid(x[:, 1 : 5, :, :]) # (B, 4, 8, 8) - bbox params
        class_probas = F.softmax(x[:, 5 : , :, :], 1) # (B, 4, 8, 8) - classes parametrization

        return torch.concat((confidence, bbox, class_probas), 1)
    
