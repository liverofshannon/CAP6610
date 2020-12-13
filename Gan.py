import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision.utils import save_image

batchSize = 100
epochs = 40
lr = 0.0002
idim = 100


class Generator(nn.Module):
    def __init__(self, idim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(idim, 56 * 56)
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 50, 3, stride=1, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(25, 1, 2, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 1, 56, 56)
        x = self.br(x)
        x = self.conv1(x)
        x = self.conv2(x)
        output = self.conv3(x)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1, padding=2),
            nn.LeakyReLU(0.2,True)
        )
        self.pl1 = nn.AvgPool2d(2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.LeakyReLU(0.2,True)
        )
        self.pl2 = nn.AvgPool2d(2, stride=2)
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 7 * 7, 1024),
            nn.LeakyReLU(0.2,True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pl1(x)
        x = self.conv2(x)
        x = self.pl2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        output = self.fc2(x)
        return output


def Gtrain(idim):
    Goptimizer.zero_grad()

    noise = torch.randn(batchSize, idim).to(device)
    realLabel = torch.ones(batchSize).to(device)
    fakeImage = G(noise)
    Doutput = D(fakeImage)
    Gloss = criterion(Doutput, realLabel)

    Gloss.backward()
    Goptimizer.step()

    return Gloss.data.item()


def Dtrain(real_img, idim):
    Doptimizer.zero_grad()

    realLabel = torch.ones(real_img.shape[0]).to(device)
    Doutput = D(real_img)
    Dreal_loss = criterion(Doutput, realLabel)

    noise = torch.randn(batchSize, idim, requires_grad=False).to(device)
    fake_label = torch.zeros(batchSize).to(device)
    fakeImage = G(noise)
    Doutput = D(fakeImage)
    Dfake_loss = criterion(Doutput, fake_label)

    Dloss = Dreal_loss + Dfake_loss

    Dloss.backward()
    Doptimizer.step()

    return Dloss.data.item()


def imgSaving(img, imGname):
    img = 0.5 * (img + 1)
    img = img.clamp(0, 1)
    save_image(img, "./imgs/" + imGname)


if __name__ == "__main__":

    device = torch.device('cpu')

    # load data
    dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=torchvision.transforms.ToTensor(),
                             download=True)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batchSize, shuffle=True)

    if not os.path.exists("./checkpoint"):
        os.makedirs("./checkpoint")

    if not os.path.exists("./imgs"):
        os.makedirs("./imgs")


    # construct generator and discriminator
    if os.path.exists('./checkpoint/Generator.pkl') and os.path.exists('./checkpoint/Discriminator.pkl'):
        G=torch.load("./checkpoint/Generator.pkl").to(device)
        D=torch.load("./checkpoint/Discriminator.pkl").to(device)
    else:
        G = Generator(idim).to(device)
        D = Discriminator().to(device)


    criterion = nn.BCELoss()
    Goptimizer = optim.Adam(G.parameters(), lr=lr)
    Doptimizer = optim.Adam(D.parameters(), lr=lr)

    print("Training begin --------------------------------------------------------------------")
    for epoch in range(1, epochs + 1):
        print("Epoch", epoch)
        for batch, (x, _) in enumerate(loader):

            # train Discriminator and generator
            Dloss=Dtrain(x.to(device), idim)
            Gloss=Gtrain(idim)


            print("[ %d / %d ]  Gloss: %.6f  Dloss: %.6f" % (batch, 600, float(Gloss), float(Dloss)))

            if batch % 50 == 0:
                fakeImage = torch.randn(128, idim)
                fakeImage = G(fakeImage)
                imgSaving(fakeImage, "imG" + str(epoch) + "_" + str(batch) + ".png")
                # save the model
                torch.save(G, "./checkpoint/Generator.pkl")
                torch.save(D, "./checkpoint/Discriminator.pkl")

