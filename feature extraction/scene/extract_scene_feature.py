# '''

#
# '''
import torch.nn as nn
import h5py
import numpy as np
from SE_resnet_conv import se_resnet50_conv
#from resnet18_conv_layer import resnet18_conv
#from resnet50_conv_layer import resnet_50
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from matplotlib import image as II
import cv2
import torch
import os


# 定义读取文件的格式
def default_loader(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image
    # return II.imread(path)



class MyDataset(Dataset):
    def __init__(self, txt, transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        f = open(txt, 'r')
        imgs = []
        line = f.readline()
        while line:
            a = line.split()
            line = f.readline()

            imgs.append((a[0], int(a[1])))
            # imgs.append((a[0]))




        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        wenjian = fn.split('\\')
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
train_img_path = r'J:\geo_affective\cross_validation\txt\scene\site_train2_16.txt'
# train_img_path = r'E:\codes\preprocess\feature extraction\new_site\txt\new_skeleton\site_train_skel1228.txt'
test_img_path = r'J:\geo_affective\cross_validation\txt\scene\site_test2_16.txt'
# val_img_path = r'E:\codes\preprocess\feature extraction\txt\skeleton\site_val_skel_new.txt'

# 数据集加载方式设置
train_data = MyDataset(txt=train_img_path, transform=transform)
train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False)
test_data = MyDataset(txt=test_img_path, transform=transform)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
# val_data = MyDataset(txt=val_img_path, transform=transform)
# val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False)


class Extract_model(nn.Module):

    def __init__(self):
        super(Extract_model, self).__init__()
        self.conv = se_resnet50_conv
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048,1024)


    def forward(self, x):
        x = self.conv(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x



extract_feature = Extract_model().cuda()

# 加载训练好的特征提取网络参数i
pretrained_dict = torch.load('J:/geo_affective/preprocess/new_site/weights/cross/scene/2/new_site_face_ep168_6147.pth')


model_dict = extract_feature.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
extract_feature.load_state_dict(model_dict)

to_org_path = r'J:\geo_affective\cross_validation\features\cross2\scene'

# h5f_data = h5py.File('features/scene_senet50_train1.h5', 'w')
# i = 0
# feature = []
# label_list = []
for data in train_loader:
    img, label, img_path = data
    path=img_path[0]
    name = path.split('\\')
    save_folder = to_org_path + os.sep + name[4][:-1]+ os.sep +name[5]+ os.sep +name[6][:-4]
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_feature_path =  save_folder+ os.sep +name[6][:-4]+'.npy'
    print(path)
    img = img.cuda()
    output = extract_feature(img).detach()
    output = output.cpu().numpy().reshape(-1)
    # feature.append(output.cpu().numpy().reshape(-1))
    # label_list.append(label.cpu().item())

    np.save(save_feature_path, output)
print("train features have been extract!")
#
#
for data in test_loader:
    img, label, img_path = data
    path=img_path[0]
    name = path.split('\\')
    save_folder = to_org_path + os.sep + name[4][:-1]+ os.sep +name[5]+ os.sep +name[6][:-4]
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_feature_path =  save_folder+ os.sep +name[6][:-4]+'.npy'
    print(path)
    img = img.cuda()
    output = extract_feature(img).detach()
    output = output.cpu().numpy().reshape(-1)
    # feature.append(output.cpu().numpy().reshape(-1))
    # label_list.append(label.cpu().item())

    np.save(save_feature_path, output)

# for data in val_loader:
#     img, label, img_path = data
#     path=img_path[0]
#     name = path.split('\\')
#     save_folder = to_org_path + os.sep + name[5]+ os.sep +name[6]+ os.sep +name[7][:-4]
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)
#     save_feature_path =  save_folder+ os.sep +name[7][:-4]+'.npy'
#     print(path)
#     img = img.cuda()
#     output = extract_feature(img).detach()
#     output = output.cpu().numpy().reshape(-1)
#     # feature.append(output.cpu().numpy().reshape(-1))
#     # label_list.append(label.cpu().item())
#
#     np.save(save_feature_path, output)
print("train features have been extract!")
# h5f_data.close()
print("train features have been extract!")

