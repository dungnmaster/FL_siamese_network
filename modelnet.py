import torch
import torch.nn as nn
import torch.nn.functional as F


class siamese_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.out = nn.Linear(256, 1)

    def hidden_forward(self, x):
        x = F.relu(self.l1(x))
        x = self.bn1(x)
        return x

    def forward(self, x1, x2):
        x = torch.abs(x1.view(-1, 512) - x2.view(-1, 512))
        return torch.sigmoid(self.out(self.hidden_forward(x)))

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive