import torch.nn as nn
import h5py
import numpy as np
from SE_resnet_conv import se_resnet50_conv
# from resnet18_conv_layer import resnet18_conv
#from resnet50_conv_layer import resnet_50
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from matplotlib import image as II
import cv2
import torch
import os


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
        # wenjian = fn.split('\\')
        img = self.loader(fn)
        img = cv2.resize(img, (224, 224))
        if self.transform is not None:
            img = self.transform(img)
        return img, label, fn

    def __len__(self):
        return len(self.imgs)


transform = transforms.Compose(
    [transforms.ToTensor()  # 函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255.
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

# class MyDataset(Dataset):
#     def __init__(self, txt, transform=None, loader=default_loader):
#         super(MyDataset, self).__init__()
#         self.data = self.read_input_file(txt)
#         self.loader = loader
#         # line = self.data[0].split(' ')
#     def read_input_file(self, file):
#         return [line.rstrip('\n') for line in open(file, encoding='UTF-8')]
#
#     def __getitem__(self, index):
#         if index >= len(self.data):
#             raise IndexError('Index out of bound')
#         sample = self.data[index].split(' ')
#         toTensor = transforms.ToTensor()
#         normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         for i in range(len(sample)-1):
#             img = cv2.imread(sample[i])
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = cv2.resize(img, (224, 224))
#             img = normalize(toTensor(img))
#             if i == 0:
#                 multi_object = img
#             if i>0 and i<16:
#                 multi_object = torch.cat([multi_object, img], dim=0)
#         label = int(sample[16])
#         # print(multi_object.shape)
#
#         return multi_object, label, sample
#
#     def __len__(self):
#         return len(self.data)


transform = transforms.Compose(
    [transforms.ToTensor(),  # 函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255.
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

train_img_path = r'J:\geo_affective\cross_validation\txt\object\site_train1_16.txt'
test_img_path = r'J:\geo_affective\cross_validation\txt\object\site_test1_16.txt'

# 数据集加载方式设置
train_data = MyDataset(txt=train_img_path, transform=transform)
train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False)
test_data = MyDataset(txt=test_img_path, transform=transform)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)


class Extract_model(nn.Module):

    def __init__(self):
        super(Extract_model, self).__init__()
        self.conv = se_resnet50_conv
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 1024)



    def forward(self, x):
        x = self.conv(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


extract_feature = Extract_model().cuda()

# 加载训练好的特征提取网络参数i
pretrained_dict = torch.load('J:/geo_affective/preprocess/new_site/weights/cross/object/cross1/new_Site_object_ep15_5282.pth')
model_dict = extract_feature.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
extract_feature.load_state_dict(model_dict)

to_org_path = r'J:\geo_affective\cross_validation\features\cross2\object'
img_id = [[] for index in range(16)]
# path = [[] for index in range(16)]

# h5f_data = h5py.File('features/object16_senet152_train.h5', 'w')
for data in train_loader:
    img, label, img_path = data
    # img_id[0],img_id[1],img_id[2],img_id[3],img_id[4],img_id[5],img_id[6],img_id[7],img_id[8],img_id[9],img_id[10],img_id[11],img_id[12],img_id[13],img_id[14],img_id[15] = img.chunk(16,dim = 1)
    # a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p = img.chunk(16,dim = 1)
    # for i in range(16):
    path = str(img_path[0])
    name = path.split('\\')
    save_folder = to_org_path + os.sep + name[5][:-1]+ os.sep +name[6] + os.sep + name[7]
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_feature_path = save_folder + os.sep + name[8][:-4] + '.npy'
    print(path)
    img = img.cuda()
    output = extract_feature(img).detach()
    output = output.cpu().numpy().reshape(-1)
    # feature.append(output.cpu().numpy().reshape(-1))
    # label_list.append(label.cpu().item())

    np.save(save_feature_path, output)

#     if i%16 == 0:
#         name = str(name).split()
#         group = h5f_data.create_group(name[5]+ '_'+name[6]+ '_'+ name[7])
#         group.create_dataset('feature', data=(np.array(feature)).astype(np.float64))
#         group.create_dataset('label', data=(np.array(label_list)).astype(np.int64))
#         feature = []
#         label_list = []
# h5f_data.close()
# print("train features have been extract!")

# h5f_data = h5py.File('features/object16_senet152_test.h5', 'w')
for data in test_loader:
    img, label, img_path = data
    # img_id[0],img_id[1],img_id[2],img_id[3],img_id[4],img_id[5],img_id[6],img_id[7],img_id[8],img_id[9],img_id[10],img_id[11],img_id[12],img_id[13],img_id[14],img_id[15] = img.chunk(16,dim = 1)
    # a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p = img.chunk(16,dim = 1)
    # for i in range(16):
    path = str(img_path[0])
    name = path.split('\\')
    save_folder = to_org_path + os.sep + name[5][:-1] + os.sep + name[6] + os.sep + name[7]
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_feature_path = save_folder + os.sep + name[8][:-4] + '.npy'
    print(path)
    img = img.cuda()
    output = extract_feature(img).detach()
    output = output.cpu().numpy().reshape(-1)
    # feature.append(output.cpu().numpy().reshape(-1))
    # label_list.append(label.cpu().item())

    np.save(save_feature_path, output)

print("test features have been extract!")