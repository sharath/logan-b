import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, nc=1, ld=100, bidirectional=True):
        super(Discriminator, self).__init__()
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.seq_z = nn.Sequential(
                nn.Flatten(3),
                
                nn.ConvTranspose2d(ld, 512, 2, 2, 0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.ConvTranspose2d(512, 512, 2, 2, 0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
            )
        
        self.seq_x = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            #nn.Conv2d(512, 1024, 4, 2, 1),
            #nn.BatchNorm2d(1024),
            #nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.seq_xz = nn.Sequential(
            nn.Conv2d(512, 1, 4),
            nn.Flatten(),
            nn.Sigmoid()
        )
    def forward(self, x, z=None):
        x = self.seq_x(x)
        if not self.bidirectional:
            return self.seq_xz(x)
        z = self.seq_z(z)
        return self.seq_xz(x+z)
    
    
class Generator(nn.Module):
    def __init__(self, nc=1, ld=100):
        super(Generator, self).__init__()
        self.seq = nn.Sequential(
            #nn.ConvTranspose2d(ld, 1024, 4, 2, 0),
            #nn.BatchNorm2d(1024),
            #nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(ld, 512, 4, 2, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, nc, 4, 2, 1),
            nn.Tanh()
        )
    def forward(self, x):
        return self.seq(x)
    
    
class Encoder(nn.Module):
    def __init__(self, nc=1, ld=100):
        super(Encoder, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            #nn.Conv2d(512, 1024, 4, 2, 1),
            #nn.BatchNorm2d(1024),
            #nn.ReLU(inplace=True),
            
            #nn.Conv2d(1024, ld, 4),
            nn.Conv2d(512, ld, 4),
            nn.Tanh()
        )
    def forward(self, x):
        return self.seq(x)