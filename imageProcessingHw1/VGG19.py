import torch.nn.functional as F
import torch.nn as nn

class VGG19(nn.Module):
    def __init__(self, num_classes):
        super(VGG19, self).__init__()

        self.block1 = nn.Sequential(nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = 1),
                                    nn.BatchNorm2d(num_features = 64),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = 1),
                                    nn.BatchNorm2d(num_features = 64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)))
        
        self.block2 = nn.Sequential(nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3, 3), stride = (1, 1), padding = 1),
                                    nn.BatchNorm2d(num_features = 128),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3, 3), stride = (1, 1), padding = 1),
                                    nn.BatchNorm2d(num_features = 128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)))
        
        self.block3 = nn.Sequential(nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (3, 3), stride = (1, 1), padding = 1),
                                    nn.BatchNorm2d(num_features = 256),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3, 3), stride = (1, 1), padding = 1),
                                    nn.BatchNorm2d(num_features = 256),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3, 3), stride = (1, 1), padding = 1),
                                    nn.BatchNorm2d(num_features = 256),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3, 3), stride = (1, 1), padding = 1),
                                    nn.BatchNorm2d(num_features = 256),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)))
        
        self.block4 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = (3, 3), stride = (1, 1), padding = 1),
                                    nn.BatchNorm2d(num_features = 512),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3, 3), stride = (1, 1), padding = 1),
                                    nn.BatchNorm2d(num_features = 512),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3, 3), stride = (1, 1), padding = 1),
                                    nn.BatchNorm2d(num_features = 512),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3, 3), stride = (1, 1), padding = 1),
                                    nn.BatchNorm2d(num_features = 512),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)))
        
        self.block5 = nn.Sequential(nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3, 3), stride = (1, 1), padding = 1),
                                    nn.BatchNorm2d(num_features = 512),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3, 3), stride = (1, 1), padding = 1),
                                    nn.BatchNorm2d(num_features = 512),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3, 3), stride = (1, 1), padding = 1),
                                    nn.BatchNorm2d(num_features = 512),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3, 3), stride = (1, 1), padding = 1),
                                    nn.BatchNorm2d(num_features = 512),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)),
                                    nn.AdaptiveAvgPool2d(7))
        
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(512 * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0, 5),
                                        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0, 5),
                                        nn.Linear(4096, num_classes), nn.Softmax(dim = 1))
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        logits = self.classifier(x.view(-1, 512 * 7 * 7))
        probas = F.softmax(logits, dim = 1)
        return logits, probas

