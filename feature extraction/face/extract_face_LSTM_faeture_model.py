
from torch import optim
import time
import torch.optim
import torch.nn as nn
from torch.autograd import Variable
from load_feature_sequence import train_loader
from load_feature_sequence import test_loader
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def plot_plot(train_acc, train_loss):
    plt.plot(range(1, len(train_acc) + 1), train_acc, 'b', label='train_acc')
    plt.legend()
    plt.show()
    plt.plot(range(1, len(train_loss) + 1), train_loss, 'r', label='train_loss')
    plt.legend()
    plt.show()


class LSTM_Model(nn.Module):

    def __init__(self):
        super(LSTM_Model, self).__init__()
        self.rnn = nn.LSTM(input_size=256, hidden_size=512, num_layers=1, batch_first=True)        #hidden_size  =  1024 为 0.691  :STM_weight2.pth
        self.fc1 = nn.Linear(512, 256)
        self.rel = nn.ReLU()
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        x = r_out[:, -1, :]
        # x = self.fc1(r_out[:, -1, :])
        x = self.fc1(x)
        x = self.rel(x)
        x = self.fc2(x)
        return x


lstm_model = LSTM_Model().cuda()
# lstm_model.load_state_dict(torch.load('model_weights/LSTM_weight_4_img.pth'))
learning_rate = 5e-6        # 1e-6 最高
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)

train_loss, train_acc = [], []
Max_acc = 0
num_epoches = 1000
for epoch in range(num_epoches):
    print(time.asctime())
    #训练
    running_loss, count, acc = 0., 0, 0.
    for step, data in enumerate(train_loader):
        img, label = data
        img = Variable(img).cuda()
        label = Variable(label).cuda()
        output = lstm_model(img)
        optimizer.zero_grad()
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        acc += (torch.max(output, dim=1)[1] == label).sum()
        count += img.size(0)
        if step % 50 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                    epoch + 1, num_epoches, running_loss / count, int(acc) / count))
    print(epoch, count, "Train_acc:", int(acc) / count, "Train_loss:", running_loss/count)

    train_acc.append(int(acc)/count)
    train_loss.append(running_loss/count)

    test_count, test_acc = 0, 0.
    for data in test_loader:
        img, label = data
        img = Variable(img).cuda()
        label = Variable(label).cuda()
        output = lstm_model(img)
        test_acc += (torch.max(output, dim=1)[1] == label).sum()
        test_count += img.size(0)
    print(test_count, "  Test_acc：",  int(test_acc) / test_count)
    if Max_acc < int(test_acc) / test_count:
        Max_acc = int(test_acc) / test_count
        torch.save(lstm_model.state_dict(), 'J:/geo_affective/preprocess/new_site/weights/cross/LSTM/2/new_site_LSTM_{}_{}.pth'.format(epoch,int(test_acc) / test_count))
    print("Max_acc:", Max_acc)
# plot_plot(train_acc, train_loss)

