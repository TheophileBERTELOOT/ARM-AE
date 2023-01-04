import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader


class AutoEncoder(nn.Module):
    def __init__(self,dataSize):
        super(AutoEncoder, self).__init__()
        self.dataSize = dataSize
        outputLayer = nn.Tanh()
        self.encoder = nn.Sequential(
            nn.Linear(self.dataSize, self.dataSize),
            outputLayer,
            nn.Linear(self.dataSize, self.dataSize),
            outputLayer,
            nn.Linear(self.dataSize, self.dataSize),
            outputLayer,
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.dataSize, self.dataSize),
            outputLayer,
            nn.Linear(self.dataSize, self.dataSize),
            outputLayer,
            nn.Linear(self.dataSize, self.dataSize),
            outputLayer,
        )

    def save(self,p):
        torch.save(self.encoder.state_dict(), p+'encoder.pt')
        torch.save(self.decoder.state_dict(),p+'decoder.pt')

    def load(self,p):
        self.encoder.load_state_dict(torch.load(p+'encoder.pt'))
        self.decoder.load_state_dict(torch.load(p + 'decoder.pt'))
        self.encoder.eval()
        self.decoder.eval()

    def forward(self, x):
        x = torch.tensor(x[None, :])
        x = self.encoder(x)
        x = self.decoder(x)
        return x






