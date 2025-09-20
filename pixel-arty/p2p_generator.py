import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# import generatora z pix2pix.py
from pix2pix import UNetGenerator  

# -------------- Funkcje pomocnicze -----------------
def load_generator(model_path, device):
    model = UNetGenerator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def load_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),   
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

    plt.imshow(fake_y)
    plt.axis("off")
    
    # żeby zapisać jako obraz 8-bitowy
    # fake_y = ((fake_y + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    # plt.savefig(save_path)
    plt.show()

# -------------- Główne wywołanie -----------------
if __name__ == "__main__":
    generate_image("generator_pix2pix.pth", r"C:\Users\rondo\Desktop\nowy_img.png", r"C:\Users\rondo\Desktop\wynik.png")
