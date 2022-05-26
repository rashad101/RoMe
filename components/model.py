import torch
import torch.nn as nn
torch.manual_seed(123)

class ScorerNN(nn.Module):
    def __init__(self, feat_size=3, hidden1=30, hidden2=20, drop1=0.3, drop2=0.2):
        super(ScorerNN, self).__init__()
        self.hid1 = nn.Linear(feat_size, hidden1)
        self.drop1 = nn.Dropout(drop1)
        self.hid2 = nn.Linear(hidden1, hidden2)
        self.drop2 = nn.Dropout(drop2)
        self.out = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.drop1(self.hid1(x))
        h2 = self.relu(self.drop2(self.hid2(h1)))
        out = self.out(h2)
        return out