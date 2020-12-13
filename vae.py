import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image


sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


imageSize = 784
hDim = 400
zDim = 20
epochNum = 40
batch_size = 128
learning_rate = 1e-3


dataSet = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)


dataLoader = torch.utils.data.DataLoader(dataset=dataSet, batch_size=batch_size, shuffle=True)


# VAE
class VAE(nn.Module):
    def __init__(self, hDim=400, zDim=20, imageSize=784):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(imageSize, hDim)
        self.fc2 = nn.Linear(hDim, zDim)
        self.fc3 = nn.Linear(hDim, zDim)
        self.fc4 = nn.Linear(zDim, hDim)
        self.fc5 = nn.Linear(hDim, imageSize)

    # encode, get mean vector and standard deviation
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    # get latent z according to mean and standard deviation ifx~N(mu, var*var), then (x-mu)/var=z~N(0, 1)
    def reparam(self, mu, logVariance):
        std = torch.exp(logVariance / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    # decode latent z
    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))

    # compute latent z
    def forward(self, x):
        mu, logVariance = self.encode(x)
        z = self.reparam(mu, logVariance)
        x_reconst = self.decode(z)
        return x_reconst, mu, logVariance


device = torch.device('cpu')

model = VAE().to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochNum):
    for i, (x, _) in enumerate(dataLoader):

        x = x.to(device).view(-1, imageSize)
        x_reconst, mu, log_var = model(x)


        # compute reconstruction loss and KL divergence
        reLoss = F.binary_cross_entropy(x_reconst, x, size_average=False)

        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Optimization
        loss = reLoss + kl_div

        optimizer.zero_grad()

        # back propagation
        loss.backward()

        optimizer.step()

        if (i + 1) % 10 == 0:
            print("epoch[{}/{}], step [{}/{}], Reconstruct Loss = {:.4f}, KL Divergence = {:.4f}"
                  .format(epoch + 1, epochNum, i + 1, len(dataLoader), reLoss.item(), kl_div.item()))

    with torch.no_grad():

        # get latent z
        z = torch.randn(batch_size, zDim).to(device)  # z的大小为batch_size * z_dim = 128*20

        out = model.decode(z).view(-1, 1, 28, 28)

        save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch + 1)))

        out, _, _ = model(x)

        x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
        save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch + 1)))