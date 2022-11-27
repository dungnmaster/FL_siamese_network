import cv2
import os
import torch
import torch.utils.data as data
import torchvision.transforms as T
from torch.utils.data import DataLoader,Dataset
import random
from PIL import Image
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import numpy as np
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.multiprocessing
from facenet_pytorch import MTCNN, InceptionResnetV1

download = True
sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)

def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)

class Config():
    path = "./data/train/"
    training_dir = "./data/train/"
    testing_dir = "./data/train/"
    validation_dir = "./data/train/"
    batch_size = 32
    train_number_epochs = 100

# class DataSource(object):
#     def __init__(self):
#         raise NotImplementedError()
#     def partitioned_by_rows(self, num_workers, test_reserve=.3):
#         raise NotImplementedError()
#     def sample_single_non_iid(self, weight=None):
#         raise NotImplementedError()

# # You may want to have IID or non-IID setting based on number of your peers 
# # by default, this code brings all dataset
# class MedMNIST(DataSource):

#     def __init__(self):
#         self.data_flag = 'pathmnist' 
#         info = INFO[self.data_flag]
#         self.n_channels = info['n_channels']
#         self.n_classes = len(info['label'])
#         self.task = info['task']

#         DataClass = getattr(medmnist, info['python_class'])

#         # preprocessing
#         data_transform = transforms.Compose([
#                          transforms.ToTensor(),
#                          transforms.Normalize(mean=[.5], std=[.5])
# 	])

#         # load the data
#         train_dataset = DataClass(split='train', transform=data_transform, download=download)
#         test_dataset = DataClass(split='test', transform=data_transform, download=download)

#         self.pil_dataset = DataClass(split='train', download=download)


#         # encapsulate data into dataloader form
#         self.train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#         self.valid_loader = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
#         self.test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

#         print(train_dataset)
#         print("===================")
#         print(test_dataset)


class DataSetFactory:

    def __init__(self, type=None):

        self.split_type = 'iid' if type == None else type
        img = np.load(Config.path + "olivetti_faces.npy") 
        transforms = T.Compose([T.ToTensor()])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        resnet = InceptionResnetV1(pretrained="vggface2").to(device).eval()
        train_data = []
        test_data = []
        validate_data = []
        # for i in range(25):
        #     train_data.append(img[i * 10 : i * 10 + 10])
        train_data = self.split_train_data(self.split_type, img, 25)
        for i in range(25,30):
            validate_data.append(img[i * 10 : i * 10 + 10])
        for i in range(30,40):
            test_data.append(img[i * 10 : i * 10 + 10])


        # make it 500 100 200
        self.num_train = 200
        self.num_valid = 10
        self.num_test = 30

        print('training size %d :validate size %d : test size %d' % (
            self.num_train, self.num_valid, self.num_test))

        training = olivetti_faces_dataset(images=train_data, transforms=transforms, size=self.num_train, resnet=resnet, device=device)
        test = olivetti_faces_dataset(images=test_data, transforms=transforms, size=self.num_test, resnet=resnet, device=device)
        validate = olivetti_faces_dataset(images=validate_data, transforms=transforms, size=self.num_valid, resnet=resnet, device=device)

        self.train_loader = DataLoader(training, batch_size=Config.batch_size, shuffle=True, worker_init_fn=set_worker_sharing_strategy)
        self.valid_loader = DataLoader(validate, batch_size=Config.batch_size, shuffle=True, worker_init_fn=set_worker_sharing_strategy)
        self.test_loader = DataLoader(test, batch_size=Config.batch_size, shuffle=True, worker_init_fn=set_worker_sharing_strategy)

    def split_train_data(self, type, img, max_range):
        data = []
        if(type == 'iid'):
            first = random.randrange(0, max_range)
            second = first
            third = first
            while(second == first):
                second = random.randrange(0, max_range)
            while(third == first):
                third = random.randrange(0, max_range)
            data.append(img[first * 10 : first * 10 + 10])
            data.append(img[second * 10 : second * 10 + 10])
            data.append(img[third * 10 : third * 10 + 10])
        else:
            first = random.randrange(0, max_range)
            second = first
            while(second == first):
                second = random.randrange(0, max_range)
            data.append(img[first * 10 : first * 10 + 10])
            data[0][0] = img[second][2]
            data[0][1] = img[second][7]
            random.shuffle(data[0])
        return data
            

class olivetti_faces_dataset(Dataset):
    def __init__(self, images, transforms, size, resnet, device):
        self.data = images
        self.transforms = transforms
        self.size = size
        self.resnet = resnet
        self.device = device

    def __len__(self):
        return self.size

    def embedding(self, x):
        with torch.no_grad():
            x = cv2.merge((x, x, x)) * 255
            x = cv2.resize(x, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
            x = (x - 127.5) / 128.0
        return self.resnet(self.transforms(x).unsqueeze(0).to(self.device))

    def __getitem__(self, idx):
        img1, img2, label = None, None, None
        if idx % 2 == 0:  # same img
            label = 1
            n = random.randrange(0, len(self.data))
            img1 = self.data[n][random.randrange(0, 10)]
            img2 = self.data[n][random.randrange(0, 10)]

        else:
            label = 0
            n = random.randrange(0, len(self.data))
            img1 = self.data[n][random.randrange(0, 10)]
            n = random.randrange(0, len(self.data))
            img2 = self.data[n][random.randrange(0, 10)]

        img1_embedding = self.embedding(img1)
        img2_embedding = self.embedding(img2)
        return (
            img1_embedding,
            img2_embedding,
            torch.FloatTensor([label]).to(self.device),
        )

if __name__ == "__main__":
    m = DataSetFactory()
    # for i,data in enumerate(m.train_loader,0):
    #     print (data[2].size(0))
    #     # print(i)
    #     # img0, img1 , label = data
    #     # print(type(img0),type(img1), type(label))
    #     # print(label)

