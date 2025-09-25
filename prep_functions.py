from PIL import Image
import numpy as np
import os

def prep_data(folder_name, start, stop=None):
    data_array = []
    folder_files = sorted(os.listdir(folder_name))

    for filename in folder_files[start:stop]:
        img_path = os.path.join(folder_name, filename)
        try:
            # RGBA aby zachować przezroczystość
            img = Image.open(img_path).convert("RGBA")
        except Exception as e:
            print(f"[prep_data] Pomijam plik {img_path}: {e}")
            continue

        pixels = np.array(img).astype(np.float32) / 255.0  # [0,1], shape (H,W,4)
        pixels = (pixels - 0.5) / 0.5
        data_array.append(pixels)

    return data_array

def give_label(numpy_array, label):
    labeled_data = [(array, label) for array in numpy_array]
    return labeled_data