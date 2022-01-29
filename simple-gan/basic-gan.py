import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


# Generator
class Generator(nn.Module):
    def __init__(self, z_dims, img_dims):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dims, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dims),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_dims):
        super().__init__()
        self.dis = nn.Sequential(
            nn.Linear(img_dims, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.dis(x)


# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
z_dims = 64
img_dims = 28 * 28 * 1
batch_size = 32
num_epochs = 200
lr = 1e-5


disc = Discriminator(img_dims).to(device)
gen = Generator(z_dims, img_dims).to(device)
fixed_noise = torch.randn(batch_size, z_dims, device=device)


transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root='data/', download=True, transform=transforms)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

optimizer_gen = optim.Adam(gen.parameters(), lr=lr)
optimizer_dis = optim.Adam(disc.parameters(), lr=lr)
criteon = nn.BCEWithLogitsLoss()

writer_fake = SummaryWriter(f'runs/GAN_MNIST/fake_images')
writer_real = SummaryWriter(f'runs/GANS_MNIST/real_images')

step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.size(0)

        ### Train Discriminator: max log(D(real)) + max log(1 - D(G(z)))

        noise = torch.randn(size=(batch_size, z_dims)).to(device)
        fake = gen(noise)
        
        disc_real = disc(real).view(-1)
        disc_real_loss = criteon(disc_real, torch.ones_like(disc_real))

        disc_fake = disc(fake.detach()).view(-1)
        disc_fake_loss = criteon(disc_fake, torch.zeros_like(disc_fake))

        disc_loss = (disc_real_loss + disc_fake_loss) / 2

        disc.zero_grad()
        disc_loss.backward()
        optimizer_dis.step()

        ### Train Generator: max log(1 - D(G(z))) <--> max log(D(G(z)))
        output = disc(fake).view(-1)
        gen_loss = criteon(output, torch.ones_like(output))
        gen.zero_grad()
        gen_loss.backward()
        optimizer_gen.step()

        if batch_idx == 0:
            print(f'Epoch: {epoch}/{num_epochs} | Discriminator Loss: {disc_loss} | Generator Loss: {gen_loss}')

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake images", img_grid_fake, global_step=step
                )

                writer_real.add_image(
                    "Mnist Real images", img_grid_real, global_step=step
                )

                step +=1



