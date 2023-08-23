import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapesDetector(torch.nn.Module):

    def __init__(self) -> None:
        super(ShapesDetector, self).__init__()
        self.relu = nn.LeakyReLU(0.01)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv1 = nn.Sequential(*[
            nn.Conv2d(3, 64, 7, 2, 3)
            ,self.relu
            ,self.pool
        ])

        self.conv2_3 = nn.Sequential(*[
            nn.Conv2d(64, 32, 3, 2, 1)
            ,self.relu
        ])
        
        self.conv2_5 = nn.Sequential(*[
            nn.Conv2d(64, 32, 5, 2, 2)
            ,self.relu
        ])
        
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(64, 64, 3, 1, 1)
            ,self.relu
            ,self.pool
        ])
        
        self.conv4 = nn.Conv2d(64, 9, 1, 1, 0)

    def forward(self, x):
        # x: (B, 3, 256, 256)
        x = self.conv1(x)
        #(B, 64, 64, 64)
        x1 = self.conv2_5(x)
        x2 = self.conv2_3(x)
        x = torch.concat((x1, x2), dim=1)
        x = self.pool(x)
        #(B, 64, 16, 16)
        x = self.conv3(x)
        #(B, 64, 8, 8)
        x = self.conv4(x)
        #(B, 9, 8, 8)
        #here the channels (0 - 4) a correspond to bounding box and confidence
        #probability and its parameters (x, y, w, h) in [0, 1]
        #channels (5-9) correspond to 4 classes probabilities
        confidence = torch.sigmoid(x[:, 0, :, :]).unsqueeze(1) #object existanse confidence
        bbox = torch.sigmoid(x[:, 1 : 5, :, :]) # (B, 4, 8, 8) - bbox params
        class_probas = F.softmax(x[:, 5 : , :, :], 1) # (B, 4, 8, 8) - classes parametrization

        return torch.concat((confidence, bbox, class_probas), 1)
    
