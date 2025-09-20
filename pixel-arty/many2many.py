import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from dataset_class import PixelDataset
import random
import matplotlib.pyplot as plt

# --- Konfiguracja ---
POSE_FOLDERS = ["pose_A", "pose_B"]  # Dodaj kolejne foldery z pozami
POSE_LABELS = list(range(len(POSE_FOLDERS)))
BATCH_SIZE = 8
IMG_SIZE = 64  # lub inny rozmiar
EPOCHS = 1  # Zwiększ do trenowania
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset rozszerzony o losowanie pozy docelowej ---
class ManyToManyDataset(torch.utils.data.Dataset):
	def __init__(self, base_dataset, num_poses):
		self.base_dataset = base_dataset
		self.num_poses = num_poses

	def __len__(self):
		return len(self.base_dataset)

	def __getitem__(self, idx):
		img, src_label = self.base_dataset[idx]
		# Losuj label docelowy różny od źródłowego
		possible_targets = [l for l in range(self.num_poses) if l != src_label.item()]
		tgt_label = random.choice(possible_targets)
		tgt_label_oh = torch.zeros(self.num_poses)
		tgt_label_oh[tgt_label] = 1.0
		return img, src_label, tgt_label_oh, tgt_label

# --- Prosty generator i dyskryminator (do rozbudowy) ---
class Generator(nn.Module):
	def __init__(self, img_channels, num_poses):
		super().__init__()
		self.fc = nn.Linear(num_poses, IMG_SIZE*IMG_SIZE)
		self.conv = nn.Sequential(
			nn.Conv2d(img_channels+1, 64, 3, padding=1),
			nn.ReLU(),
			nn.Conv2d(64, img_channels, 3, padding=1),
			nn.Tanh()
		)

	def forward(self, x, pose_label):
		# pose_label: one-hot, shape [batch, num_poses]
		b, c, h, w = x.shape
		pose_map = pose_label.view(b, -1, 1, 1).expand(-1, -1, h, w).sum(1, keepdim=True)
		x_cat = torch.cat([x, pose_map], dim=1)
		return self.conv(x_cat)

class Discriminator(nn.Module):
	def __init__(self, img_channels, num_poses):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(img_channels, 64, 3, padding=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(64, 1, 3, padding=1)
		)
		self.classifier = nn.Linear(IMG_SIZE*IMG_SIZE, num_poses)

	def forward(self, x):
		feat = self.conv(x)
		out_adv = feat.mean([2,3])  # global avg pooling
		out_cls = self.classifier(feat.view(feat.size(0), -1))
		return out_adv, out_cls

# --- Przygotowanie danych ---
base_dataset = PixelDataset(folder_names=POSE_FOLDERS, labels=POSE_LABELS)
dataset = ManyToManyDataset(base_dataset, num_poses=len(POSE_FOLDERS))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Inicjalizacja modeli ---
generator = Generator(img_channels=3, num_poses=len(POSE_FOLDERS)).to(DEVICE)
discriminator = Discriminator(img_channels=3, num_poses=len(POSE_FOLDERS)).to(DEVICE)

# --- Pętla treningowa (szkic) ---
for epoch in range(EPOCHS):
	for imgs, src_labels, tgt_labels_oh, tgt_labels in dataloader:
		imgs = imgs.to(DEVICE)
		tgt_labels_oh = tgt_labels_oh.to(DEVICE)
		# --- Generator ---
		fake_imgs = generator(imgs, tgt_labels_oh)
		# --- Dyskryminator ---
		out_adv, out_cls = discriminator(fake_imgs)
		# ...oblicz straty, optymalizuj, itd.
		# ...zapisz modele, generuj wyniki, itd.

# --- Generowanie wszystkich poz dla jednego obrazka ---
def generate_all_poses(img, generator, num_poses):
	results = []
	img = img.unsqueeze(0).to(DEVICE)
	for pose in range(num_poses):
		pose_oh = torch.zeros(1, num_poses).to(DEVICE)
		pose_oh[0, pose] = 1.0
		gen_img = generator(img, pose_oh)
		results.append(gen_img.cpu().detach())
	return results


# --- Wizualizacja wyników ---
def show_generated_poses(img, generator, num_poses, pose_names=None):
	"""
	img: tensor [3, H, W] w zakresie [-1,1]
	generator: model
	num_poses: liczba możliwych pozycji
	pose_names: opcjonalnie lista nazw pozycji
	"""
	gen_imgs = generate_all_poses(img, generator, num_poses)
	fig, axs = plt.subplots(1, num_poses+1, figsize=(3*(num_poses+1), 3))
	# Oryginał
	axs[0].imshow(((img.cpu().permute(1,2,0)+1)/2).numpy())
	axs[0].set_title("Oryginał")
	axs[0].axis("off")
	# Wygenerowane pozy
	for i, gen_img in enumerate(gen_imgs):
		axs[i+1].imshow(((gen_img[0].cpu().permute(1,2,0)+1)/2).numpy())
		if pose_names:
			axs[i+1].set_title(f"Poza {pose_names[i]}")
		else:
			axs[i+1].set_title(f"Poza {i}")
		axs[i+1].axis("off")
	plt.show(block=False)
	plt.pause(5)
	plt.close()
    
# --- Przykład użycia po epokach ---
if __name__ == "__main__":
	# Pobierz przykładowy obrazek z datasetu
	sample_img, _, _, _ = dataset[0]
	show_generated_poses(sample_img, generator, len(POSE_FOLDERS), pose_names=POSE_FOLDERS)
