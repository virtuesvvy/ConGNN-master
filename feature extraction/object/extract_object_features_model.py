from load_imgs_object import test_loader,train_loader,num_of_trainData
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import matplotlib.pyplot as plt
# from vgg_net import
from SE_resnet_conv import se_resnet50_conv
import time
# from resnet18_conv_layer import resnet18_conv
# from resnet50_conv_layer import resnet_50
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Extract_model(nn.Module):

    def __init__(self):
        super(Extract_model, self).__init__()
        self.conv = se_resnet50_conv
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 1024)
        self.rel = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 3)
        # self.rel = nn.ReLU(inplace=True)
        # self.fc3 = nn.Linear(256, 3)


    def forward(self, x):
        x = self.conv(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.rel(x)
        x = self.fc2(x)
        # x = self.rel(x)
        # x = self.fc3(x)

        return x


extract_model = Extract_model().cuda()


# pretrained_dict = torch.load('model_weights/object16_senet152_57.72.pth')
# model_dict = extract_model.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# extract_model.load_state_dict(model_dict)

learning_rate = 1e-4
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(extract_model.parameters(), lr=learning_rate)
max_acc = 0.0
train_loss, train_acc = [], []
num_epoches = 5000
for epoch in range(num_epoches):
    print(time.asctime())
    aa = 0
    #训练
    running_loss, count, acc = 0., 0, 0.
    for step, data in enumerate(train_loader):
        img, label = data
        img = Variable(img).cuda()
        label = Variable(label).cuda()
        output = extract_model(img)
        optimizer.zero_grad()
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        acc += (torch.max(output, dim=1)[1] == label).sum()
        count += img.size(0)
        if step % 100 == 0 and step > 0:
            print('{}, [{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                    epoch + 1, count, num_of_trainData, running_loss / count, int(acc) / count))
    print(epoch, count, "Train_acc:", int(acc) / count, "Train_loss:", running_loss/count)
    #保存模型参数
    #torch.save(extract_model.state_dict(), 'model_weights/extract_feature_model' + str(epoch) + '.pth')

    test_count, test_acc = 0, 0.
    for data in test_loader:
        img, label = data
        img = Variable(img).cuda()

        label = Variable(label).cuda()
        output = extract_model(img)
        test_acc += (torch.max(output, dim=1)[1] == label).sum()
        test_count += img.size(0)
    print(test_count, "  Test_acc：",  int(test_acc) / test_count)
    if max_acc < int(test_acc) / test_count:
        max_acc = int(test_acc) / test_count
        torch.save(extract_model.state_dict(), 'J:/geo_affective/preprocess/new_site/weights/cross/object/cross1/new_site_object_ep{}_{}.pth'.format(epoch,int(test_acc) / test_count))
    print("max_acc:", max_acc)