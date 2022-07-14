import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader


class AutoEncoder(nn.Module):
    def __init__(self,dataSize,baseline,lstm,gru,vae,conv,conv2D,dropout,hiddenSize,output):
        super(AutoEncoder, self).__init__()
        self.baseline = baseline
        self.lstm = lstm
        self.gru = gru
        self.vae = vae
        self.conv = conv
        self.conv2D = conv2D
        self.dataSize = dataSize
        self.hiddenSize = hiddenSize
        self.output = output
        self.weight = torch.FloatTensor(self.dataSize, self.dataSize)
        if output == 'relu':
            outputLayer = nn.ReLU(True)
        if output == 'softmax':
            outputLayer = nn.Softmax(dim=1)
        if output == 'sigmoid':
            outputLayer = nn.Sigmoid()
        if output == 'tanh':
            outputLayer = nn.Tanh()
        if lstm:
            hiddenLayer = nn.LSTM(
                input_size=self.dataSize,    # 45, see the data definition
                hidden_size=self.hiddenSize,
                num_layers=2,
                dropout=dropout
            )
        if gru:
            hiddenLayer = nn.GRU(
                input_size=self.dataSize,  # 45, see the data definition
                hidden_size=self.hiddenSize,
                num_layers=2,
                dropout=dropout
            )
        if baseline:
            self.encoder = nn.Sequential(
                nn.Linear(self.dataSize, self.hiddenSize),
                outputLayer,
                nn.Linear(self.dataSize, self.hiddenSize),
                outputLayer,
                nn.Linear(self.dataSize, self.hiddenSize),
                outputLayer,
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.hiddenSize, self.dataSize),
                outputLayer,
                nn.Linear(self.dataSize, self.hiddenSize),
                outputLayer,
                nn.Linear(self.dataSize, self.hiddenSize),
                outputLayer,
            )
        elif conv:
            self.conv1 = nn.Conv1d(self.dataSize, 16, 3,padding=1)
            self.conv2 = nn.Conv1d(16,32, 3, padding=1)
            # Decoder
            self.t_conv1 = nn.ConvTranspose1d(32, 16, 3, stride=1,padding=1)
            self.t_conv2 = nn.ConvTranspose1d(16, self.dataSize, 3, stride=1,padding=1)
        elif conv2D:
            self.conv1 = nn.Conv2d(1, 16, 5,padding=1)
            self.conv2 = nn.Conv2d(16,32, 3, padding=1)
            # Decoder
            self.t_conv1 = nn.ConvTranspose2d(32, 16, 3, stride=1,padding=1)
            self.t_conv2 = nn.ConvTranspose2d(16, 1, 5, stride=1,padding=1)
            self.linear1Enc = nn.Linear(self.dataSize, self.hiddenSize)
            self.linear2Enc = nn.Linear(self.hiddenSize, self.hiddenSize)
            self.linear3Enc = nn.Linear(self.hiddenSize, self.hiddenSize)
            self.linear1Dec = nn.Linear(self.hiddenSize, self.hiddenSize)
        elif vae:
            self.N = torch.distributions.Normal(0, 1)
            self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()
            self.kl = 0
            self.linear1Enc = nn.Linear(self.dataSize, self.hiddenSize)
            self.linear2Enc = nn.Linear(self.hiddenSize, self.hiddenSize)
            self.linear3Enc = nn.Linear(self.hiddenSize, self.hiddenSize)
            self.linear1Dec = nn.Linear(self.hiddenSize, self.hiddenSize)

        else:
            self.encoder = nn.ModuleDict({
            'hidden': hiddenLayer,
            'linear': nn.Linear(
                in_features=self.hiddenSize,
                out_features=self.hiddenSize),
            'output':outputLayer

        })
            self.decoder = nn.ModuleDict({
            'hidden': hiddenLayer,
            'linear': nn.Linear(
                in_features=self.hiddenSize,
                out_features=self.dataSize),
            'output': outputLayer

        })





    def forward(self, x):
        if self.baseline:
            x = torch.tensor(x[None, :])
            x = self.encoder(x)
            x = self.decoder(x)
        elif self.vae:
            x = torch.flatten(x, start_dim=1)
            x = F.relu(self.linear1Enc(x))
            mu = self.linear2Enc(x)
            sigma = torch.exp(self.linear3Enc(x))
            z = mu + sigma * self.N.sample(mu.shape)
            self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
            x = F.relu(self.linear1Dec(z))
        elif self.conv:
            x = x.reshape((1,self.dataSize,self.dataSize))
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.t_conv1(x))
            x = F.relu(self.t_conv2(x))
        elif self.conv2D:
            x = x.reshape((1,1,self.dataSize,self.dataSize))
            x = F.tanh(self.conv1(x))
            x = F.tanh(self.conv2(x))
            x = F.tanh(self.t_conv1(x))
            x = F.tanh(self.t_conv2(x))
        else:
            x = torch.tensor(x[None, :])
            x,_ = self.encoder['hidden'](x)
            x = self.encoder['linear'](x)
            x = self.encoder['output'](x)
            x,_ = self.decoder['hidden'](x)
            x = self.decoder['linear'](x)
            x = self.decoder['output'](x)
        return x


    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    m.weight.data.fill_(1)



