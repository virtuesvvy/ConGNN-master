import torch.nn as nn
import h5py
import numpy as np
#from model_framework.SE_resnet_conv import se_resnet50_conv
# from resnet18_conv_layer import resnet18_conv
from resnet50_conv_layer import resnet_50
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from matplotlib import image as II
import cv2
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 定义读取文件的格式
def default_loader(path):
    return II.imread(path)


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        f = open(txt, 'r')
        imgs = []
        line = f.readline()
        while line:
            a = line.split()
            line = f.readline()

            imgs.append((a[0], int(a[16])))
            imgs.append((a[1], int(a[16])))
            imgs.append((a[2], int(a[16])))
            imgs.append((a[3], int(a[16])))
            imgs.append((a[4], int(a[16])))
            imgs.append((a[5], int(a[16])))
            imgs.append((a[6], int(a[16])))
            imgs.append((a[7], int(a[16])))
            imgs.append((a[8], int(a[16])))
            imgs.append((a[9], int(a[16])))
            imgs.append((a[10], int(a[16])))
            imgs.append((a[11], int(a[16])))
            imgs.append((a[12], int(a[16])))
            imgs.append((a[13], int(a[16])))
            imgs.append((a[14], int(a[16])))
            imgs.append((a[15], int(a[16])))

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        wenjian = fn.split('\\')
        img = self.loader(fn)
        img = cv2.resize(img, (112, 112))
        if self.transform is not None:
            img = self.transform(img)
        return img, label, wenjian

    def __len__(self):
        return len(self.imgs)


transform = transforms.Compose(
    [transforms.ToTensor()  # 函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255.
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

train_img_path = r'J:\geo_affective\cross_validation\txt\face\site_train2_16.txt'
test_img_path = r'J:\geo_affective\cross_validation\txt\face\site_test2_16.txt'

# 数据集加载方式设置
train_data = MyDataset(txt=train_img_path, transform=transform)
train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False)
test_data = MyDataset(txt=test_img_path, transform=transform)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)


class Extract_model(nn.Module):

    def __init__(self):
        super(Extract_model, self).__init__()
        self.conv = resnet_50
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048,1024)
        self.rel = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 256)


    def forward(self, x):
        x = self.conv(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.rel(x)
        x = self.fc2(x)

        return x


extract_feature = Extract_model().cuda()

# 加载训练好的特征提取网络参数i
pretrained_dict = torch.load('J:/geo_affective/preprocess/new_site/weights/cross/face/2/new_site_face_ep88_6813.pth')
model_dict = extract_feature.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
extract_feature.load_state_dict(model_dict)

h5f_data = h5py.File('J:/geo_affective/cross_validation/features/cross2/face/site_face_train530.h5', 'w')
i = 0
feature = []
label_list = []
for data in train_loader:
    i += 1
    img, label, wenjian = data
    img = img.cuda()
    label = label.cuda()
    output = extract_feature(img).detach()
    feature.append(output.cpu().numpy().reshape(-1))
    label_list.append(label.cpu().item())

    xx = np.array(wenjian[5] + wenjian[6] + wenjian[7])

    if i % 16 == 0:
        print(xx)
        group = h5f_data.create_group(xx[0]+'_'+xx[1]+'_'+xx[2])
        group.create_dataset('feature', data=(np.array(feature)).astype(np.float64))
        group.create_dataset('label', data=(np.array(label_list)).astype(np.int64))
        feature = []
        label_list = []
h5f_data.close()
print("train features have been extract!")


h5f_data = h5py.File('J:/geo_affective/cross_validation/features/cross2/face/site_face_test530.h5', 'w')
i = 0
feature = []
label_list = []
for data in test_loader:
    i += 1
    img, label, wenjian = data
    img = img.cuda()
    label = label.cuda()
    output = extract_feature(img).detach()
    feature.append(output.cpu().numpy().reshape(-1))
    label_list.append(label.cpu().item())

    xx = np.array(wenjian[5] + wenjian[6] + wenjian[7])

    if i % 16 == 0:
        print(xx)
        group = h5f_data.create_group(xx[0]+'_'+xx[1]+'_'+xx[2])
        group.create_dataset('feature', data=(np.array(feature)).astype(np.float64))
        group.create_dataset('label', data=(np.array(label_list)).astype(np.int64))
        feature = []
        label_list = []
h5f_data.close()
print("test features have been extract!")