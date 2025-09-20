import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import torch.autograd as autograd
import matplotlib.pyplot as plt
import main
from PIL import Image
import os


# hiperparametry
latent_dim = 100         # rozmiar noise wektora
batch_size = 64
lr = 1e-4
n_show = 250          # co ile epok wyświetlamy wygenerowane obrazy
n_critic = 5             # co ile generacji trenujemy krytyka
n_epochs = 1000
lambda_gp = 10           # gradient penalty
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("CUDA available:", torch.cuda.is_available())
# generator -----------------------------------------------------------------------------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3*48*48),
            nn.Tanh()  # output [-1, 1]
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(z.size(0), 3, 48, 48)
        return img

# krytyk -----------------------------------------------------------------------------------------------------------------------------------------
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*48*48, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, img):
        return self.model(img)

# gradient penalty --------------------------------------------------------------------------------------------------------------------------------
def compute_gradient_penalty(critic, real, fake):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolates = critic(interpolates)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


generator = Generator(latent_dim).to(device)
critic = Critic().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
optimizer_C = optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))

# pętla -------------------------------------------------------------------------------------------------------------------------------------------
for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(main.dataloader):
        imgs = imgs.to(device)

        # ---- Train Critic ----
        z = torch.randn(imgs.size(0), latent_dim, device=device)
        fake_imgs = generator(z)

        critic_real = critic(imgs)
        critic_fake = critic(fake_imgs)
        gp = compute_gradient_penalty(critic, imgs, fake_imgs)

        loss_C = -(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * gp

        optimizer_C.zero_grad()
        loss_C.backward()
        optimizer_C.step()


        if i % n_critic == 0:
            z = torch.randn(imgs.size(0), latent_dim, device=device)
            fake_imgs = generator(z)
            loss_G = -torch.mean(critic(fake_imgs))

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()


    print(f"[Epoch {epoch+1}/{n_epochs}] Loss_C: {loss_C.item():.4f} Loss_G: {loss_G.item():.4f}")
    if (epoch+1) % n_show == 0:
        generator.eval()
        with torch.no_grad():
            z = torch.randn(16, latent_dim, device=device)
            sample_imgs = generator(z).cpu()
            # z [-1,1] do [0,255] 
            sample_imgs = ((sample_imgs * 0.5 + 0.5) * 255).clamp(0, 255).byte()    
        fig, axs = plt.subplots(4, 4, figsize=(6,6))
        for idx, ax in enumerate(axs.flatten()):
            img = sample_imgs[idx].permute(1,2,0).numpy()
            ax.imshow(img)
            ax.axis('off')
        plt.show(block=False)
        plt.pause(5)
        generator.train()



# Save the trained generator ---------------------------------------------------------------------------------------------------------------------
# torch.save(generator.state_dict(), "generator.pth")   zapiszemy jak będzie działać

'''
zapisanie do plików
generator.eval()
with torch.no_grad():
    z = torch.randn(16, latent_dim, device=device)
    sample_imgs = generator(z).cpu()
    sample_imgs = ((sample_imgs * 0.5 + 0.5) * 255).clamp(0, 255).byte()

    os.makedirs("generated_images", exist_ok=True)
    for idx, img_tensor in enumerate(sample_imgs):
        img = img_tensor.permute(1, 2, 0).numpy()  
        img_pil = Image.fromarray(img)
        img_pil.save(f"generated_images/image_{idx:02d}.png")
'''

generator.eval()
with torch.no_grad():
    z = torch.randn(16, latent_dim, device=device)
    sample_imgs = generator(z).cpu()
    # z [-1,1] do [0,255] 
    sample_imgs = ((sample_imgs * 0.5 + 0.5) * 255).clamp(0, 255).byte()    
fig, axs = plt.subplots(4, 4, figsize=(6,6))
for idx, ax in enumerate(axs.flatten()):
    img = sample_imgs[idx].permute(1,2,0).numpy()
    ax.imshow(img)
    ax.axis('off')
plt.show()
