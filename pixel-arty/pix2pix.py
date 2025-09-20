import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
import matplotlib.pyplot as plt
from prep_functions import prep_data
# ===================== dataset  =========================

class PairedPixelDataset(torch.utils.data.Dataset):
    def __init__(self, folder_0, folder_1, start=0, stop=None):
        self.data_0 = prep_data(folder_0, start, stop)
        self.data_1 = prep_data(folder_1, start, stop)
        assert len(self.data_0) == len(self.data_1)

    def __len__(self):
        return len(self.data_0)

    def __getitem__(self, idx):
        img0 = torch.tensor(self.data_0[idx], dtype=torch.float).permute(2,0,1)
        img1 = torch.tensor(self.data_1[idx], dtype=torch.float).permute(2,0,1)
        return img0, img1
    
# ===================== Generator  =========================
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),  
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),         
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),        
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),         
            nn.ReLU(True),
            nn.Dropout(0.5),

        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), 
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1), 
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),  
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, out_channels, 4, 2, 1), 
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x


# ===================== Dyskryminator =========================
class Discriminator(nn.Module):
    def __init__(self, in_channels=6):  # input + target
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),  
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),        
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),       
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 4, 1, 1)          
        )

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)  
        return self.model(inp)

#  ===================== Hiperparametry =========================

n_epochs = 10
batch_size = 64
lr = 2e-4
start = 0       # start i stop to zakres indeksów obrazków w folderach
stop = 10000
n_show = 50     # co ile epok wyświetlamy wygenerowane obrazy

# ===================== Pętla =========================
def train_pix2pix(folder_0, folder_1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PairedPixelDataset(folder_0, folder_1, start=start, stop=stop)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    generator = UNetGenerator().to(device)
    discriminator = Discriminator().to(device)

    opt_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    adv_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    for epoch in range(n_epochs):
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            #  train discriminator 
            fake_y = generator(x).detach()
            real_validity = discriminator(x, y)
            fake_validity = discriminator(x, fake_y)

            real_loss = adv_loss(real_validity, torch.ones_like(real_validity))
            fake_loss = adv_loss(fake_validity, torch.zeros_like(fake_validity))
            d_loss = (real_loss + fake_loss) / 2

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            #  train generator 
            fake_y = generator(x)
            fake_validity = discriminator(x, fake_y)
            g_adv = adv_loss(fake_validity, torch.ones_like(fake_validity))
            g_l1 = l1_loss(fake_y, y) * 100
            g_loss = g_adv + g_l1

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

        print(f"[{epoch+1}/{n_epochs}] D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

        if (epoch+1) % n_show == 0:
            generator.eval()
            with torch.no_grad():
                sample_fake = generator(x[:4]).cpu()
            generator.train()

            fig, axs = plt.subplots(3, 4, figsize=(10, 7))
            for j in range(4):
                axs[0, j].imshow(((x[j].cpu().permute(1,2,0)+1)/2).numpy())
                axs[0, j].set_title(f"Input:"+f"{folder_0[-3:]}")
                axs[1, j].imshow(((y[j].cpu().permute(1,2,0)+1)/2).numpy())
                axs[1, j].set_title(f"Input:"+f"{folder_1[-3:]}")
                axs[2, j].imshow(((sample_fake[j].cpu().permute(1,2,0)+1)/2).numpy())
                axs[2, j].set_title("Generated")
                for k in range(3): axs[k, j].axis("off")
            plt.show(block=False)
            plt.pause(5)
        
        if (epoch+1) == n_epochs:
            generator.eval()
            with torch.no_grad():
                sample_fake = generator(x[:4]).cpu()
            generator.train()

            fig, axs = plt.subplots(3, 4, figsize=(10, 7))
            for j in range(4):
                axs[0, j].imshow(((x[j].cpu().permute(1,2,0)+1)/2).numpy())
                axs[0, j].set_title(f"Input:"+f"{folder_0[-3:]}")
                axs[1, j].imshow(((y[j].cpu().permute(1,2,0)+1)/2).numpy())
                axs[1, j].set_title(f"Input:"+f"{folder_1[-3:]}")
                axs[2, j].imshow(((sample_fake[j].cpu().permute(1,2,0)+1)/2).numpy())
                axs[2, j].set_title("Generated")
                for k in range(3): axs[k, j].axis("off")
            plt.show()
    torch.save(generator.state_dict(), "generator_pix2pix.pth")



from dotenv import load_dotenv
load_dotenv()
path_1 = os.getenv("PATH_1")
path_2 = os.getenv("PATH_2")






if __name__ == "__main__":
    train_pix2pix(
        path_1,
        path_2       
    )










