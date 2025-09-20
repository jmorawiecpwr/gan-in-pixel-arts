from PIL import Image
import numpy as np
import os

def prep_data(folder_name,start,stop=None):

    data_array = []
    folder_files = os.listdir(folder_name)
    
    for filename in folder_files[start:stop]:
        img_path = os.path.join(folder_name, filename)
        img = Image.open(img_path).convert("RGB")
        pixels = np.array(img)
        img_float = pixels.astype(np.float32) / 255.0
        data_array.append(img_float)
    
    return data_array

def give_label(numpy_array,label):
    labeled_data = [(array, label) for array in numpy_array]
    return labeled_data