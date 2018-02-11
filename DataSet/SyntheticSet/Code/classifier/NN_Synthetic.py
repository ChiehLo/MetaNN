"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.1.11
matplotlib
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as Data


torch.manual_seed(1)    # reproducible

'''
# make fake data
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer

# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()
'''

BATCH_SIZE = 32
EPOCH = 25
LR = 0.005
#LR = 0.01
#LR = 0.01

input_dim = 100
h1_dim = 512
h2_dim = 256
h3_dim = 512
h4_dim = 64     
output_dim = 8
accuracy_sum = 0

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_hidden2, n_hidden3, n_hidden4, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden2)   # hidden layer
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)   # hidden layer
        #self.hidden4 = torch.nn.Linear(n_hidden3, n_hidden4)   # hidden layer
        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.normalB = torch.nn.BatchNorm1d(n_feature)
        self.out = torch.nn.Linear(n_hidden2, n_output)   # output layer

    def forward(self, x):
        #x = self.normalB(x)
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.dropout1(x)
        x = F.relu(self.hidden2(x))
        #x = F.relu(self.hidden3(x))
        #x = F.relu(self.hidden4(x))
        x = self.dropout1(x)
        x = self.out(x)
        return x

p1 = np.array([0.1])
p2 = np.array([0.1])
p3 = np.array([0.9])
i1 = 0
i2 = 0
i3 = 0
postfix = '_' + str(p1[i1]) + '_' + str(p2[i2]) + '_' + str(p3[i3])
#postfix = ''
print(postfix)


nsplits = 10
for ii in range(nsplits):
    it = ii +1
    train_sample_path = './Synthetic/IT' + str(it) + '/TrainSampleSynthetic_' + str(it) + postfix + '.txt'
    train_label_path = './Synthetic/IT' + str(it) + '/TrainLabelSynthetic_' + str(it) + postfix + '.txt'
    train_sample = np.loadtxt(train_sample_path, delimiter='\t')
    train_label = np.loadtxt(train_label_path, delimiter='\t')
    row_sums = train_sample.sum(axis=1)
    row_sums = row_sums[:, np.newaxis]
    row_sums[np.where(row_sums == 0.0)] = 1 
    train_sample = train_sample / row_sums
    sample_tensor = torch.from_numpy(train_sample).type(torch.FloatTensor)
    label_tensor = torch.from_numpy(train_label).type(torch.LongTensor)


    #print(sample_tensor)
    #print(label_tensor)

    '''
    train_sample = np.loadtxt('./T2D/IT1/nbSampleWT2D_1_1000.txt', delimiter='\t')
    train_label = np.loadtxt('./T2D/IT1/nbLabelWT2D_1_1000.txt', delimiter='\t')
    row_sums = train_sample.sum(axis=1)
    train_sample = train_sample / row_sums[:, np.newaxis]

    vae_tensor = torch.from_numpy(train_sample).type(torch.FloatTensor)
    vae_label_tensor = torch.from_numpy(train_label).type(torch.LongTensor)

    total_sample = torch.cat((sample_tensor, vae_tensor), 0)
    total_label = torch.cat((label_tensor, vae_label_tensor), 0)

    Train_dataset = torch.utils.data.TensorDataset(total_sample, total_label)
    train_loader = torch.utils.data.DataLoader(Train_dataset, shuffle=True, batch_size = BATCH_SIZE)
    '''

    Train_dataset = torch.utils.data.TensorDataset(sample_tensor, label_tensor)
    train_loader = torch.utils.data.DataLoader(Train_dataset, shuffle=True, batch_size = BATCH_SIZE)


    test_sample_path = './Synthetic/IT' + str(it) + '/TestSampleSynthetic_' + str(it) + postfix + '.txt'
    test_label_path = './Synthetic/IT' + str(it) + '/TestLabelSynthetic_' + str(it) + postfix + '.txt'

    test_sample = np.loadtxt(test_sample_path, delimiter='\t')
    test_label = np.loadtxt(test_label_path, delimiter='\t')
    row_sums = test_sample.sum(axis=1)
    row_sums = row_sums[:, np.newaxis]
    row_sums[np.where(row_sums == 0.0)] = 1 
    test_sample = test_sample / row_sums

    '''
    test_sample = np.loadtxt('./T2D/IT1/nbSampleall_1_1000.txt', delimiter='\t')
    test_label = np.loadtxt('./T2D/IT1/nbLabelall_1_1000.txt', delimiter='\t')
    row_sums = test_sample.sum(axis=1)
    test_sample = test_sample / row_sums[:, np.newaxis]
    '''
    test_tensor = torch.from_numpy(test_sample).type(torch.FloatTensor)
    test_label_tensor = torch.from_numpy(test_label).type(torch.LongTensor)

    #print(test_tensor)
    #print(test_label_tensor)


    net = Net(n_feature=input_dim, n_hidden=h1_dim, n_hidden2 = h2_dim, n_hidden3 = h3_dim, n_hidden4 = h4_dim, n_output=output_dim)     # define the network
    print(net)  # net architecture

    #ptimizer = torch.optim.SGD(net.parameters(), lr=0.02)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted



    for epoch in range(EPOCH):
        net.train()
        for step, (X, Y) in enumerate(train_loader):
            x = Variable(X)   # batch x, shape (batch, 28*28)
            Y = torch.squeeze(Y)
            #print(Y)
            y = Variable(Y)   # batch y, shape (batch, 28*28)

            out = net(x)                 # input x and predict based on x
            loss = loss_func(out, y)
        
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step()                    # apply gradients

            if step % 20 == 0:
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0])
                #plt.cla()
                prediction = torch.max(F.softmax(out), 1)[1]
                pred_y = prediction.data.numpy().squeeze()
                target_y = y.data.numpy()
                #plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
                accuracy = sum(pred_y == target_y)/BATCH_SIZE
                print(accuracy)
                #plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
                #plt.pause(0.1)
        net.eval()
        test_tensor1 = Variable(test_tensor)
        out = net(test_tensor1) 
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = test_label_tensor.numpy()
        accuracy = sum(pred_y == target_y)/target_y.shape[0]
        print(accuracy)

    #plt.ioff()
    #plt.show()
    net.eval()
    sample_tensor = Variable(sample_tensor)
    out = net(sample_tensor) 
    prediction = torch.max(F.softmax(out), 1)[1]
    pred_y = prediction.data.numpy().squeeze()
    target_y = label_tensor.squeeze().numpy()
    #print(pred_y)
    #print(target_y)
    accuracy = sum(pred_y == target_y)/target_y.shape[0]
    print(accuracy)


    test_tensor = Variable(test_tensor)
    out = net(test_tensor) 
    prediction = torch.max(F.softmax(out), 1)[1]
    pred_y = prediction.data.numpy().squeeze()
    target_y = test_label_tensor.numpy()
    accuracy = sum(pred_y == target_y)/target_y.shape[0]
    print(accuracy)
    accuracy_sum = accuracy + accuracy_sum

print(accuracy_sum/nsplits)




