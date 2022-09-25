'''
本代码 使用训练好的lstm网络提取小视频片段特征并融合，提取为视频特征图
'''

import torch.nn as nn
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os


class MyDataset(Dataset):
    def __init__(self, x, y, name):
        super(MyDataset, self).__init__()
        self.x = x
        self.y = y
        self.name = name

    def __getitem__(self, index):
        img = torch.tensor(self.x[index]).float()
        label = self.y[index]
        name = self.name[index]
        return img, label, name

    def __len__(self):
        return len(self.x)


x, y, train_name = [], [], []
f = h5py.File('J:/geo_affective/cross_validation/features/cross2/face/site_face_train530.h5','r')
for name in f:

    x.append(f[name]['feature'].value)
    label = f[name]['label'].value
    y.append(label[0].astype(np.int64))
    print(name)
    train_name.append(name)
f.close()

tx, ty, test_name = [], [], []
f = h5py.File('J:/geo_affective/cross_validation/features/cross2/face/site_face_test530.h5','r')
for name in f:
    tx.append(f[name]['feature'].value)
    label = f[name]['label'].value
    ty.append(label[0].astype(np.int64))
    test_name.append(name)
f.close()

train_set = MyDataset(x, y, train_name)
train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=False)
test_set = MyDataset(tx, ty, test_name)
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)


class LSTM_Model(nn.Module):

    def __init__(self):
        super(LSTM_Model, self).__init__()
        self.rnn = nn.LSTM(input_size=256, hidden_size=512, num_layers=1, batch_first=True)        #hidden_size  =  1024 为 0.691  :STM_weight2.pth
        self.fc1 = nn.Linear(512, 256)


    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        x = r_out[:, -1, :]
        x = self.fc1(x)
        print(x)

        return x

extract_feature = LSTM_Model().cuda()

# 加载训练好的特征提取网络参数
pretrained_dict = torch.load('J:/geo_affective/preprocess/new_site/weights/cross/LSTM/2/new_site_LSTM_36_8183.pth')
model_dict = extract_feature.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
extract_feature.load_state_dict(model_dict)


to_org_path = r"J:\geo_affective\cross_validation\features\cross2\LSTM"
# h5f_data = h5py.File('features/site_face_LSTM_train.h5', 'w')
i = 0
# feature = []
# label_list = []

for data in train_loader:
    i += 1
    img, label, name = data
    img = img.cuda()
    label = label.cuda()
    output = extract_feature(img).detach()
    output = output.cpu().numpy().reshape(-1)
    name = str(name).split('_')
    save_folder = to_org_path + os.sep + name[0][2:7] + os.sep + name[1] + os.sep + name[2][:-3]
    # save_folder = to_org_path + os.sep + name[0] + os.sep + name[2] + os.sep + name[3]
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for i in range(16):
        save_feature_path = save_folder + os.sep + str(i) + '.npy'
        np.save(save_feature_path, output)

# h5f_data.close()
print("train features have been extract!")

# h5f_data = h5py.File('features/site_face_LSTM_test.h5', 'w')
i=0
# feature = []
# label_list = []
for data in test_loader:
    i += 1
    img, label, name = data
    img = img.cuda()
    label = label.cuda()
    output = extract_feature(img).detach()
    output = output.cpu().numpy().reshape(-1)
    name = str(name).split('_')
    save_folder = to_org_path + os.sep + name[0][2:6] + os.sep + name[1] + os.sep + name[2][:-3]
    # save_folder = to_org_path + os.sep + name[0] + os.sep + name[1] + os.sep + name[2]

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for i in range(16):
        save_feature_path = save_folder + os.sep + str(i) + '.npy'
        np.save(save_feature_path, output)
# h5f_data.close()
print("test features have been extract!")