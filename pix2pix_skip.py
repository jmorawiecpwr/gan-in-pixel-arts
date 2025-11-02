import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from prep_functions import prep_data
import hiperparametry as hp

# ===================== Dataset =========================
class PairedPixelDataset(torch.utils.data.Dataset):
    def __init__(self, folder_0, folder_1, start=0, stop=None):
        self.data_0 = prep_data(folder_0, start, stop)  # (H,W,4) w [-1,1]
        self.data_1 = prep_data(folder_1, start, stop)
        assert len(self.data_0) == len(self.data_1), "Liczba obrazów w folderach nie jest taka sama." 

    def __len__(self):
        return len(self.data_0)

    def __getitem__(self, idx):
        img0 = torch.tensor(self.data_0[idx], dtype=torch.float).permute(2,0,1)
        img1 = torch.tensor(self.data_1[idx], dtype=torch.float).permute(2,0,1)
        return img0, img1

# ===================== setup =========================
n_epochs = hp.n_epochs
batch_size = hp.batch_size
lr = hp.lr
loss_level = hp.loss_level
w_decay = hp.w_decay
is_show = hp.is_show

train_start = hp.train_start
train_stop = hp.train_stop
test_start = hp.test_start
test_stop = hp.test_stop

def down_block(in_ch, out_ch, norm=True):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

def bottleneck_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(True)
    )

def up_block(in_ch, out_ch, dropout=0.0):
    layers = [
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(True)
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

# ===================== Generator =========================
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, dropout=0.5):
        super().__init__()
        # Down 48->24->12->6
        self.d1 = down_block(in_channels, 64, norm=False)  # 48->24
        self.d2 = down_block(64, 128)                      # 24->12
        self.d3 = down_block(128, 256)                     # 12->6
        # Bottleneck (nie zmniejsza spatial)
        self.bottleneck = bottleneck_block(256, 512)       # 6->6
        # Up
        self.u1 = up_block(512, 256, dropout=dropout)      # 6->12 
        self.u2 = up_block(384, 128, dropout=dropout)      # 12->24
        self.u3 = up_block(192, 64, dropout=0.0)           # 24->48
        # Wyjście (utrzymuje 48x48)
        self.out = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        d1 = self.d1(x)   # 24
        d2 = self.d2(d1)  # 12
        d3 = self.d3(d2)  # 6
        b  = self.bottleneck(d3)  # 6
        u1 = self.u1(b)           # 12
        u1 = torch.cat([u1, d2], dim=1)
        u2 = self.u2(u1)          # 24
        u2 = torch.cat([u2, d1], dim=1)
        u3 = self.u3(u2)          # 48
        return self.out(u3)

# ===================== Discriminator =========================
class Discriminator(nn.Module):
    def __init__(self, in_channels=8):  
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),  # 48 -> 24
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),         # 24 -> 12
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),        # 12 -> 6
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 4, 1, 1)           # Patch map
        )

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)
        return self.model(inp)

# ===================== Pętla treningowa =========================

def train_pix2pix(folder_0, folder_1, save_name="generator_pix2pix.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = PairedPixelDataset(folder_0, folder_1, start=train_start, stop=train_stop)
    test_dataset = PairedPixelDataset(folder_0, folder_1, start=test_start, stop=test_stop)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    generator = UNetGenerator(dropout=hp.drop).to(device)
    discriminator = Discriminator().to(device)

    opt_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=w_decay)
    opt_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=w_decay)

    adv_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    for epoch in range(n_epochs):
        generator.train(); discriminator.train()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            # --- Train D ---
            with torch.no_grad():
                fake_y = generator(x)
            real_validity = discriminator(x, y)
            fake_validity = discriminator(x, fake_y)

            real_loss = adv_loss(real_validity, torch.ones_like(real_validity))
            fake_loss = adv_loss(fake_validity, torch.zeros_like(fake_validity))
            d_loss = (real_loss + fake_loss) / 2
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()

            # --- Train G ---
            fake_y = generator(x)
            fake_validity = discriminator(x, fake_y)
            g_adv = adv_loss(fake_validity, torch.ones_like(fake_validity))
            g_l1 = l1_loss(fake_y, y) * loss_level
            g_loss = g_adv + g_l1
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()

        print(f"[Skip {epoch+1}/{n_epochs}] D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

    torch.save(generator.state_dict(), save_name)
    return save_name

if __name__ == "__main__":
    train_pix2pix(hp.path1, hp.path2)
















