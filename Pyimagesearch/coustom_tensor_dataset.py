from torch.utils.data import Dataset 

class CoustomTensorDataset(Dataset):
    def __init__(self, tensors,transform=None):
        self.tesnors=tensors 
        # tuiple of three tensors (image,label,boundbox coordinates)
        self.transform=transform

    

    def __getitem__ (self,index):
        image=self.tensors[0][index]
        label=self.tensors[1][index]
        bbox=self.tensors[2][index]

        image=image.permute(2,0,-1)

        if self.transform:
            image=self.transform(image)

        return (image,label,bbox)    
    

    def __len__(self):
        return self.tensors[0].size[0]
    



