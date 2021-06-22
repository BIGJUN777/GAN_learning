import torch
import torch.nn as nn

def block(in_dim, out_dim, normalize=True, drpp_prob=None, bias=True):
    layers = [nn.Linear(in_dim, out_dim, bias=bias)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_dim))
    layers.append(nn.LeakyReLU(0.2))
    if drpp_prob:
        layers.append(nn.Dropout(drpp_prob))
    return layers

class Generator(nn.Module):
    def __init__(self, in_dim, out_dim, num_class):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            *block(in_dim+num_class, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, out_dim),
            nn.Tanh()
        )
        self.num_class = num_class

    def forward(self, z, labels):
        if self.num_class:
            label_oh_emb = torch.zeros(z.size(0), self.num_class).scatter_(1, labels.view(-1,1), 1).to(z.device)
            z = torch.cat((z, label_oh_emb), dim=1)
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, in_dim, out_dim, num_class):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            *block(in_dim+num_class, 512, False),
            *block(512, 256, True),
            nn.Linear(256, out_dim),
            nn.Sigmoid()        
        )
        self.num_class=num_class

    def forward(self, x, labels):
        if self.num_class:
            label_oh_emb = torch.zeros(x.size(0), self.num_class).scatter_(1, labels.view(-1,1), 1).to(x.device)
            x = torch.cat((x, label_oh_emb), dim=1)
        return self.model(x)