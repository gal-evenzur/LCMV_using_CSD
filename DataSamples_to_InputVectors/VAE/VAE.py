"""NICE model
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, latent_dim,device):
        """Initialize a VAE.

        Args:
            latent_dim: dimension of embedding
            device: run on cpu or gpu
        """
        super(Model, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.criterion = nn.BCELoss(reduction='sum')
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 1, 2),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  7, 7
        )

        self.mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.logvar = nn.Linear(64 * 7 * 7, latent_dim)

        self.upsample = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  64,  14,  14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1, 1),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 1, 2),  # B, 1, 28, 28
            nn.Sigmoid()
        )


    def sample(self,sample_size,mu=None,logvar=None):
        '''
        :param sample_size: Number of samples
        :param mu: z mean, None for prior (init with zeros)
        :param logvar: z logstd, None for prior (init with zeros)
        :return:
        '''
        if mu==None:
            mu = torch.zeros((sample_size,self.latent_dim)).to(self.device)
        if logvar == None:
            logvar = torch.zeros((sample_size,self.latent_dim)).to(self.device)
        
        z = torch.randn_like(mu)
        x = self.upsample(z)
        x = x.view(-1, 64, 7, 7)
        x = self.decoder(x)
        return x

    def z_sample(self, mu, logvar):
        return mu + torch.randn_like(mu)*torch.sqrt(torch.exp(logvar))

    def loss(self,x,recon,mu,logvar):
        return self.criterion(recon, x)+0.5*(torch.exp(logvar)+mu**2-logvar-1).sum()

    def forward(self, x):
        latent_base = self.encoder(x)
        latent_base = latent_base.view(-1, 64*7*7)
        z = self.z_sample(self.mu(latent_base), self.logvar(latent_base))
        x_pred = self.upsample(z)
        x_pred = x_pred.view(-1, 64, 7, 7)
        x_pred = self.decoder(x_pred)
        elbo = self.loss(x, x_pred, self.mu(latent_base), self.logvar(latent_base))
        return elbo