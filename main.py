from dataset_class import PixelDataset
from torch.utils.data import DataLoader

# zakładam, że macie foldery z danymi
pre_dataset = PixelDataset(folder_names=["0_0","0_1"],labels=[0,1], start=0, stop=1000)
dataloader = DataLoader(pre_dataset, batch_size=64, shuffle=True)

# Wyświetlenie pierwszego batcha (przykładowo 64 elementowego)
data_iter = iter(dataloader)
images, labels = next(data_iter)

print(images.shape)  
print(labels)       