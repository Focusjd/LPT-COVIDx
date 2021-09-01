import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms.transforms import RandomRotation



ROOT = '../data'
trainfile = 'train_COVIDx_A.txt'
validatefile = 'validate_COVIDx_A.txt'
testfile = 'test_COVIDx_A.txt'
index_path = ROOT + '/index/'

class COVIDxDataset(Dataset):
    """
    Code for reading the COVIDxDataset
    """
    def __init__(self, mode, data_path=ROOT,  dim=(224, 224)):
        self.data_path = data_path
        self.root = os.path.join(data_path, mode, '')
        #  ROOT + '/' + mode + '/'

        self.dim = dim
        self.class_dict = {'pneumonia': 0, 'normal': 1, 'COVID-19': 2}
        self.CLASSES = len(self.class_dict)

        if (mode == 'train'):
            self.paths, self.labels = read_filepaths(index_path+trainfile)
            self.do_augmentation = True
        elif (mode == 'validate'):
            self.paths, self.labels = read_filepaths(index_path+validatefile)
            self.do_augmentation =  False
        elif (mode == 'test'):
            self.paths, self.labels = read_filepaths(index_path+testfile)
            self.do_augmentation =  False
        elif (mode == 'eval_train'):
            paths_train, labels_train = read_filepaths(index_path+trainfile)
            paths_validate, labels_validate = read_filepaths(index_path+validatefile)
            self.paths = paths_train + paths_validate
            self.labels = labels_train + labels_validate
            self.do_augmentation =  True

        print("{} examples =  {}".format(mode, len(self.paths)))
        self.mode = mode
        

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.mode == 'eval_train':
            if index<6500:
                image_tensor = self.load_image(os.path.join(self.data_path, 'train', '')+ self.paths[index], self.dim, augmentation=self.mode)
                label_tensor = torch.tensor(self.class_dict[self.labels[index]], dtype=torch.long)
            else:
                image_tensor = self.load_image(os.path.join(self.data_path, 'validate', '') + self.paths[index], self.dim, augmentation=self.mode)
                label_tensor = torch.tensor(self.class_dict[self.labels[index]], dtype=torch.long)
        else:    
            image_tensor = self.load_image(self.root + self.paths[index], self.dim, augmentation=self.mode)
            label_tensor = torch.tensor(self.class_dict[self.labels[index]], dtype=torch.long)

        return image_tensor, label_tensor

    def load_image(self, img_path, dim, augmentation='test'):
        if not os.path.exists(img_path):
            print("IMAGE DOES NOT EXIST {}".format(img_path))
        image = Image.open(img_path).convert('RGB')
        image = image.resize(dim)

        if self.do_augmentation:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        image_tensor = transform(image)

        return image_tensor


def read_filepaths(file):
    paths, labels = [], []
    with open(file, 'r') as f:
        lines = f.read().splitlines()

        for i, line in enumerate(lines):

            path, label = line.split(' ')[1], line.split(' ')[2]
 
            paths.append(path)
            labels.append(label)
    return paths, labels

def main():
    batch_size = 32

    train_data = COVIDxDataset(mode='train')
    valid_data = COVIDxDataset(mode='test')

    train_queue = DataLoader(train_data, batch_size=batch_size, pin_memory=False, num_workers=0)
    valid_queue = DataLoader(valid_data, batch_size=batch_size, pin_memory=False, num_workers=0)

    # img1, label1 = train_data[13830]
    # img2, label2 = train_data[900]
    # print(label1 , label2)
    # print(type(train_queue))
    # print(next(iter(train_queue))[1])

    # train_features, train_labels = next(iter(train_queue))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")

    train_features = train_data[9]
    img = train_features[0].squeeze().numpy().transpose(1,2,0)
    print(img.shape)
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    main()