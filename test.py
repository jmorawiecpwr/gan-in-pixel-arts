import torch
class dairedPixelDataset(torch.utils.data.Dataset):
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
print(dairedPixelDataset)