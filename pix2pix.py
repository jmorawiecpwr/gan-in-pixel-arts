import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
import matplotlib.pyplot as plt
from prep_functions import prep_data
import hiperparametry as hp
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

# ===================== Train/Val Split =========================
full_size = hp.full_size
num_holdout = hp.num_holdout
test_split = hp.test_split
usable_size = hp.usable_size
test_size = hp.test_size
train_size = hp.train_size
train_start = hp.train_start
train_stop = hp.train_stop
test_start = hp.test_start
test_stop = hp.test_stop
    
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
            nn.ReLU(True),)
        
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
n_epochs = hp.n_epochs
batch_size = hp.batch_size
lr = hp.lr
loss_level = hp.loss_level
w_decay = hp.w_decay
is_show = hp.is_show
# ===================== Pętla =========================
def train_pix2pix(folder_0, folder_1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train/test split
    train_dataset = PairedPixelDataset(folder_0, folder_1, start=train_start, stop=train_stop)
    test_dataset = PairedPixelDataset(folder_0, folder_1, start=test_start, stop=test_stop)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    generator = UNetGenerator().to(device)
    discriminator = Discriminator().to(device)

    opt_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=w_decay)
    opt_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=w_decay)

    # opt_G = optim.RMSprop(generator.parameters(), lr=lr, weight_decay=w_decay)
    # opt_D = optim.RMSprop(discriminator.parameters(), lr=lr, weight_decay=w_decay)

    adv_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    for epoch in range(n_epochs):
        for i, (x, y) in enumerate(train_loader):
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
            g_l1 = l1_loss(fake_y, y) * loss_level
            g_loss = g_adv + g_l1

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

        print(f"[{epoch+1}/{n_epochs}] D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

        # Show test results
        generator.eval()
        with torch.no_grad():
            test_x, test_y = next(iter(test_loader))
            test_x, test_y = test_x.to(device), test_y.to(device)
            test_fake = generator(test_x).cpu()
        generator.train()

        if (epoch+1) == n_epochs and is_show == True:
            fig, axs = plt.subplots(3, 4, figsize=(10, 7))
            for j in range(4):
                axs[0, j].imshow(((test_x[j].cpu().permute(1,2,0)+1)/2).numpy())
                axs[0, j].set_title(f"Test Input: {os.path.basename(folder_0)}")
                axs[1, j].imshow(((test_y[j].cpu().permute(1,2,0)+1)/2).numpy())
                axs[1, j].set_title(f"Test Target: {os.path.basename(folder_1)}")
                axs[2, j].imshow(((test_fake[j].cpu().permute(1,2,0)+1)/2).numpy())
                axs[2, j].set_title("Generated")
                for k in range(3): axs[k, j].axis("off")
            plt.show(block=False)
            if (epoch+1) == n_epochs:
                plt.show()
            else:
                plt.pause(5)
    torch.save(generator.state_dict(), "generator_pix2pix.pth")

    
if __name__ == "__main__":
    train_pix2pix(
        r"C:\Users\RODO\Desktop\gan-in-pixel-arts\0_0",
        r"C:\Users\RODO\Desktop\gan-in-pixel-arts\0_1"       
    )










