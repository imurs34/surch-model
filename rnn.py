import torch
import torch.nn as nn
from torchvision import models
import torch.nn.init as init


class RNNModel(nn.Module):
    def __init__(self, n_classes=8):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        self.fc = nn.Linear(512, n_classes)
        self.dropout = nn.Dropout(p=0.2)

        init.xavier_normal_(self.lstm.all_weights[0][0])
        init.xavier_normal_(self.lstm.all_weights[0][1])
        init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        seq_len = x.shape[1]

        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, seq_len, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)
        y = self.dropout(y)
        y = self.fc(y)
        return y


class Ensemble:
    def __init__(self, *models):
        self.models = models

    def __call__(self, x, *args):
        p = [model(x, *args) for model in self.models]
        p = torch.stack(p, dim=-1)
        p = torch.mean(p, dim=-1)
        return p

    def eval(self):
        for model in self.models:
            model.eval()

    def train(self):
        for model in self.models:
            model.train()
