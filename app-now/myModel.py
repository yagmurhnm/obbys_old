
#%%
from __future__ import print_function
#%matplotlib inline
import random
import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
# Set random seed for reproducibility
#manualSeed = 999
manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
#%%
#from google.colab import drive
#drive.mount('/content/drive')
#%%
# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
#%%

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
#%%
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
netG = Generator(ngpu).to(device)
netG.load_state_dict(torch.load("C:/Users/tugcu/Desktop/this is mine/Generator_NetG.pth",map_location ='cpu'))
netG.eval()
#%%
noise = torch.randn(64, nz, 1, 1, device=device)
fake = netG(noise).detach().cpu()
#fake = netG(fixed_noise).detach().cpu()
img = vutils.make_grid(fake, padding=2, normalize=True)
#%%
plt.subplot(1,2,2)
plt.axis("off")
plt.imshow(np.transpose(img,(1,2,0)))
plt.show()

# %%
import pickle
with open('bike_model_xgboost.pkl', 'wb') as file:
    pickle.dump(classifier, file)
# %%
