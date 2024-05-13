import torchvision.models as models
import torch.nn as nn

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        
        # Load the pre-trained ResNet50 model
        resnet50 = models.resnet50()
        
        # Remove the existing fully connected layer
        self.features = nn.Sequential(*list(resnet50.children())[:-1])
        
        # Add a new fully connected layer with 1 output node
        self.fc = nn.Linear(2048, 1)
        
        # Add a Sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
