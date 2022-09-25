
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from matplotlib import image as II
import cv2
import os

from PIL import Image

# 定义读取文件的格式
def default_loader(path):
    # image = Image.open(path)
    image = II.imread(path)
    # image = cv2.imread(path)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

# 首先继承上面的dataset类。然后在__init__()方法中得到图像的路径，然后将图像路径组成一个数组，这样在__getitim__()中就可以直接读取：
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
        img = self.loader(fn)
        img = cv2.resize(img, (224, 224))
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)


transform = transforms.Compose(
    [transforms.ToTensor()  # 函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255.
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

Train_txt_path = r'J:\geo_affective\cross_validation\txt\object\site_train1_16.txt'
Val_txt_path = r'J:\geo_affective\cross_validation\txt\object\site_test1_16.txt'
# 数据集加载方式设置

train_data = MyDataset(txt=Train_txt_path, transform=transform)
test_data = MyDataset(txt=Val_txt_path, transform=transform)
train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=8, shuffle=False)
num_of_trainData = len(train_data)
print('num_of_trainData:', len(train_data))
print('num_of_testData:', len(test_data))


