import pickle as pickle
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
class RAPv2(Dataset):
    def __init__(self, dataset_path, phase, transform=None, target_transform=None):
        self.path = dataset_path
        self.set = phase
        self.transform = transform
        self.target_transform = target_transform

        self.dataset = pickle.load(open(self.path+'/rap2_dataset.pkl','rb'))
        self.partition = pickle.load(open(self.path+'/rap2_partition.pkl','rb'))
        self.att_name = [self.dataset['att_name'][i] for i in self.dataset['selected_attribute']]

        self.image = []
        self.label = []

        for idx in self.partition[self.set][0]: #self.partition['train'][0]
            self.image.append(self.dataset['image'][idx])
            self.label.append(np.array(self.dataset['att'][idx])[self.dataset['selected_attribute']])

        self.label = torch.tensor(self.label)

    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, item):
        filename, target = self.image[item], self.label[item]
        img = Image.open(self.path+'/RAP_dataset/'+filename)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        else:
            target = target*2-1

        data = {'image': img, 'name': filename, 'target': target}

        return data

    def num_att(self):
        return len(self.dataset['selected_attribute'])
