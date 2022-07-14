

import os

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np





num_epochs = 10
batch_size = 128
learning_rate = 1e-3


train = pd.read_csv('data/mushroom.csv',dtype=float)
train_tensor = torch.tensor(train.values)
dataloader = DataLoader(train_tensor.float(), batch_size=batch_size, shuffle=True)

x = torch.tensor([float('nan'), float('inf'), -float('inf'), 3.14])
torch.nan_to_num(x, nan=0.0, posinf=0.0)
print(len(train.loc[0]))



class autoencoder(nn.Module):
    def __init__(self,baseline,lstm):
        super(autoencoder, self).__init__()
        self.baseline = baseline
        self.lstm = lstm
        self.size = len(train.loc[0])
        if baseline:
            self.encoder = nn.Sequential(
                nn.Linear(self.size, self.size),
                nn.ReLU(True)
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.size, self.size),
                nn.ReLU(True))
        if lstm:
            self.encoder = self.model = nn.ModuleDict({
            'lstm': nn.LSTM(
                input_size=self.size,    # 45, see the data definition
                hidden_size=self.size,  # Can vary
            ),
            'linear': nn.Linear(
                in_features=self.size,
                out_features=self.size),

        })
            self.decoder = nn.ModuleDict({
            'lstm': nn.LSTM(
                input_size=self.size,    # 45, see the data definition
                hidden_size=self.size,  # Can vary
            ),
            'linear': nn.Linear(
                in_features=self.size,
                out_features=self.size),

        })






    def forward(self, x):
        if self.baseline:
            x = torch.tensor(x[None, :])
            x = self.encoder(x)
            x = self.decoder(x)
            return x
        if self.lstm:
            x = torch.tensor(x[None, :])
            x,_ = self.encoder['lstm'](x)
            x = self.encoder['linear'](x)
            x,_ = self.decoder['lstm'](x)
            x = self.decoder['linear'](x)
            return x


    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    m.weight.data.fill_(1)





model = autoencoder(False,True).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        d = Variable(data).cuda()
        # ===================forward=====================
        output = model.forward(d)
        loss = criterion(output, d)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data))

test = np.zeros(len(train.loc[0]))
consequent = 84
test[consequent] = 1
test = torch.tensor(test).cuda()
test = test.unsqueeze(0)
output = model(test.float())
output = np.array(output.cpu().detach().numpy())
output = pd.DataFrame(output.T.reshape(len(train.loc[0]), -1))
output = pd.DataFrame(output)
print(output)
print(output[0].nlargest(10))


antecedent = 84
suppD = train[train.columns[[consequent,antecedent]]]
suppD = sum([sum(suppD.loc[x]) == 2 for x in range(len(suppD))])/len(suppD)
print('support:'+str(suppD))
antD =train[train.columns[[antecedent]]].sum()[0]/len(train)
confD = suppD/(antD)
print('confiance:'+str(confD))
cons = (train[train.columns[[consequent]]].sum()[0]/len(train))
cosD = suppD/np.sqrt(cons*antD)
print('cosine:'+str(cosD))

antecedent =89
suppD = train[train.columns[[consequent,antecedent]]]
suppD = sum([sum(suppD.loc[x]) == 2 for x in range(len(suppD))])/len(suppD)
print('support:'+str(suppD))
antD =train[train.columns[[antecedent]]].sum()[0]/len(train)
confD = suppD/(antD)
print('confiance:'+str(confD))
cons = (train[train.columns[[consequent]]].sum()[0]/len(train))
cosD = suppD/np.sqrt(cons*antD)
print('cosine:'+str(cosD))

antecedent =37
suppD = train[train.columns[[consequent,antecedent]]]
suppD = sum([sum(suppD.loc[x]) == 2 for x in range(len(suppD))])/len(suppD)
print('support:'+str(suppD))
antD =train[train.columns[[antecedent]]].sum()[0]/len(train)
confD = suppD/(antD)
print('confiance:'+str(confD))
cons = (train[train.columns[[consequent]]].sum()[0]/len(train))
cosD = suppD/np.sqrt(cons*antD)
print('cosine:'+str(cosD))





torch.save(model.state_dict(), './sim_autoencoder.pth')