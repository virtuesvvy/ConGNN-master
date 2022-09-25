import torch.optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import h5py
import numpy as np
import torch.optim
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class MyDataset(Dataset):
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        img = torch.tensor(self.x[index]).float()
        # img = torch.tensor(cv2.resize(self.x[index], (224, 224))).float().unsqueeze(0)
        label = self.y[index]
        return img, label

    def __len__(self):
        return len(self.x)


x, y = [], []
f = h5py.File('J:/geo_affective/cross_validation/features/cross2/face/site_face_train530.h5','r')
for name in f:
    x.append(f[name]['feature'][()])
    label = f[name]['label'][()]
    y.append(label[0].astype(np.int64))
f.close()

tx, ty = [], []
f = h5py.File('J:/geo_affective/cross_validation/features/cross2/face/site_face_test530.h5','r')
for name in f:
    tx.append(f[name]['feature'][()])
    label = f[name]['label'][()]
    ty.append(label[0].astype(np.int64))
f.close()


batch_size = 16

train_set = MyDataset(x, y)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_set = MyDataset(tx, ty)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
print(len(train_set))
print(len(test_set))
