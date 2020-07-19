import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models

COLORS = ['Blue', 'White', 'Red', 'Silver', 'Grey', 'Black', 'Green']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class model_color(nn.Module):
  def __init__(self):
    super(model_color, self).__init__()
    self.layers = nn.ModuleList()
    self.layers.append(models.resnet18(pretrained=True))
    self.layers.append(nn.Linear(1000, 256)) 
    self.layers.append(nn.Dropout(0.1))
    self.layers.append(nn.Linear(256, 32))
    self.layers.append(nn.Sigmoid())
    self.layers.append(nn.Dropout(0.1))
    self.layers.append(nn.Linear(32, len(COLORS)))
    self.layers.append(nn.Softmax())
    
  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x


def load_color_model():
    model = model_color().eval()
    model.load_state_dict(torch.load('models/model_color_77.pt', map_location=torch.device(device)))
    return model
