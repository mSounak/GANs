import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from model import Generator, Discriminator, initialize_weights


# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
z_dim = 100
lr = 2e-4
n_epochs = 10
features_g = 64
features_d = 64
img_size = 64
channels_img = 1


transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(channels_img)],
        [0.5 for _ in range(channels_img)]
    ),
])

# dataset
dataset = datasets.MNIST(root='data/', transform=transforms)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models
gen = Generator(z_dim, channels_img, features_g).to(device)
disc = Discriminator(channels_img, n_features=features_d).to(device)
initialize_weights(model=gen)
initialize_weights(model=disc)

# optimizer
optimizer_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

# loss
criterion = nn.BCELoss()

# fixed noise
fixed_noise = torch.randn(batch_size, z_dim, 1, 1, device=device)

# tensorboard
writer_real = SummaryWriter(f'runs/DCGAN_MNIST/real_images')
writer_fake = SummaryWriter(f'runs/DCGAN_MNIST/fake_images')

step = 0


# train
for epoch in range(n_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn(batch_size, z_dim, 1, 1, device=device)

        # train discriminator
        optimizer_disc.zero_grad()
        fake = gen(noise)
        
        disc_real = disc(real).reshape(-1)
        disc_real_loss = criterion(disc_real, torch.ones_like(disc_real))

        disc_fake = disc(fake).reshape(-1)
        disc_fake_loss = criterion(disc_fake, torch.zeros_like(disc_fake))

        disc_loss = (disc_real_loss + disc_fake_loss) / 2
        disc_loss.backward(retain_graph=True)
        optimizer_disc.step()

        # train generator
        optimizer_gen.zero_grad()
        output = disc(fake).reshape(-1)
        gen_loss = criterion(output, torch.ones_like(disc(fake).reshape(-1)))
        gen_loss.backward()
        optimizer_gen.step()

        # tensorboard
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}/{n_epochs} | Discriminator Loss: {disc_loss} | Generator Loss: {gen_loss}')

            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                writer_fake.add_image(
                    "Mnist DCGAN Fake images", img_grid_fake, global_step=step
                )

                writer_real.add_image(
                    "Mnist DCGAN Real images", img_grid_real, global_step=step
                )

                step +=1

