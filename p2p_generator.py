import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from pix2pix import UNetGenerator, train_pix2pix  
import hiperparametry as hp
# -------------- Funkcje pomocnicze -----------------
def load_generator(model_path, device):
    model = UNetGenerator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,weights_only=False))
    model.eval()
    return model

def load_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((48, 48)),   
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)  # skalowanie do [-1,1]
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)  # [1,3,H,W]

def generate_image(model_path, img_path, save_path="generated.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_generator(model_path, device)

    x = load_image(img_path).to(device)

    with torch.no_grad():
        fake_y = model(x).cpu()

    # denormalizacja z [-1,1] → [0,1]
    fake_y = (fake_y.squeeze(0).permute(1,2,0).numpy() + 1) / 2

    
    img_uint8 = (fake_y * 255).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(img_uint8)
    im.save(save_path)

    
    # plt.imshow(fake_y, interpolation='nearest')
    # plt.axis("off")
    # plt.show()

# -------------- Główne wywołanie -----------------
if __name__ == "__main__":
    # ta funkcja trenuje model
    train_pix2pix(
        hp.path1,
        hp.path2
    )
    # ta tworzy obraz z podanego pliku
    generate_image("generator_pix2pix.pth", hp.input_path, hp.output_path)
