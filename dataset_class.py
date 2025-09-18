import torch
from prep_functions import give_label,prep_data

class PixelDataset():
    
    def __init__(self,folder_name=None,folder_names=None,start=0,stop=None,label=None,labels=None):
        self.data_array = []
        self.labels = []
        
        if not folder_name and not folder_names:
            raise KeyError("You must provide str for 'folder_name' or list of names for 'folder_names'")
        
        if folder_name and folder_names:
            raise KeyError("You must provide 'folder_name' or 'folder_names'")
        
        if not label and not labels:
            raise KeyError("You must provide int for 'label' or list of ints for 'labels'")
        if label and labels:
            raise KeyError("You must provide 'label' or 'labels'")
        
        if folder_name and label:
            images = prep_data(folder_name,start,stop)
            labeled = give_label(images,label)
        
            for img_array,label in labeled:
                self.data_array.append(img_array)
                self.labels.append(label)
        
        if folder_names and labels:
            if isinstance(folder_names,list) and all(isinstance(name,str) for name in folder_names):
                if isinstance(labels,list) and all(isinstance(name,int) for name in labels):                
                    for name, lbl in zip(folder_names, labels):
                        images = prep_data(name, start, stop)
                        labeled = give_label(images, lbl)
                        
                        for img_array, tag in labeled:
                            self.data_array.append(img_array)
                            self.labels.append(tag)
                else:
                    raise KeyError("labels must be list of ints")
            else:
                raise KeyError("folder_names must be list of strings")
        else:
            raise KeyError("Provide label for every folder")
    
    
    def __len__(self):
        return len(self.data_array)
    
    def __getitem__(self,index):
        image = self.data_array[index]
        label = self.labels[index]
        
        tensor_img = torch.tensor(image,dtype=torch.float).permute(2,0,1)
        tensor_label = torch.tensor(label,dtype=torch.int64)
    
        return tensor_img,tensor_label