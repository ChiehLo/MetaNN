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


torch.manual_seed(3)    # reproducible



BATCH_SIZE = 32
EPOCH = 200
LR = 0.005


input_dim = 1025
h1_dim = 512
h2_dim = 256
h3_dim = 128
h4_dim = 64     
output_dim = 2 


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_hidden2, n_hidden3, n_hidden4, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden2)   # hidden layer
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)   # hidden layer
        self.dropout1 = torch.nn.Dropout(p=0.5)   # hidden layer
        self.out = torch.nn.Linear(n_hidden3, n_output)   # output layer
    def forward(self, x):
        x = self.hidden(x)
        x = self.dropout1(x)
        x = F.relu(x)      # activation function for hidden layer
        x = self.hidden2(x)
        #x = self.dropout1(x)
        x = F.relu(x)
        x = self.hidden3(x)
        #x = self.dropout1(x)
        x = F.relu(x)
            #x = F.relu(self.hidden4(x))
        x = self.dropout1(x)
        x = self.out(x)
        return x





IT = np.array([1, 2, 3, 4, 5])
pp = np.array([50, 75, 100, 150, 200, 500, 1000, 2000, 5000])

prefix = './IBD_data/IT'

for it in range(len(IT)):

    #### load training samples
    sample_file = prefix + str(it+1) + '/TrainSample_' + str(it+1) + '.txt'
    label_file = prefix + str(it+1) + '/TrainLabel_' + str(it+1) + '.txt'
    train_sample = np.loadtxt(sample_file, delimiter='\t')
    train_label = np.loadtxt(label_file, delimiter='\t')
    row_sums = train_sample.sum(axis=1)
    train_sample = train_sample / row_sums[:, np.newaxis]

    temp = np.zeros((len(train_label),1))

    for i in range(len(train_label)):
        for j in range(2):
            if train_label[i,j] == 1:
                temp[i] = j

    
    sample_tensor = torch.from_numpy(train_sample).type(torch.FloatTensor)
    label_tensor = torch.from_numpy(temp).type(torch.LongTensor)

    #### load testing samples
    sample_file = prefix + str(it+1) + '/TestSample_' + str(it+1) + '.txt'
    label_file = prefix + str(it+1) + '/TestLabel_' + str(it+1) + '.txt'
    test_sample = np.loadtxt(sample_file, delimiter='\t')
    test_label = np.loadtxt(label_file, delimiter='\t')
    row_sums = test_sample.sum(axis=1)
    test_sample = test_sample / row_sums[:, np.newaxis]

    if torch.cuda.is_available():
        test_tensor = torch.from_numpy(test_sample).type(torch.FloatTensor).cuda()
        test_label_tensor = torch.from_numpy(test_label).type(torch.LongTensor).cuda()
    else:
        test_tensor = torch.from_numpy(test_sample).type(torch.FloatTensor)
        test_label_tensor = torch.from_numpy(test_label).type(torch.LongTensor)

    #### load nb data
    for cc in range(len(pp)):
        sample_file = prefix + str(it+1) + '/nbSample_' + str(it+1) + '_' + str(pp[cc]) + '.txt'
        label_file = prefix + str(it+1) + '/nbLabel_' + str(it+1) + '_' + str(pp[cc]) + '.txt'
        train_sample = np.loadtxt(sample_file, delimiter='\t')
        train_label = np.loadtxt(label_file, delimiter='\t')
        row_sums = train_sample.sum(axis=1)
        train_sample = train_sample / row_sums[:, np.newaxis]
        vae_tensor = torch.from_numpy(train_sample).type(torch.FloatTensor)
        vae_label_tensor = torch.from_numpy(train_label).type(torch.LongTensor)

        total_sample = torch.cat((sample_tensor, vae_tensor), 0)
        total_label = torch.cat((label_tensor, vae_label_tensor), 0)

        print (total_sample)
        print(total_label)

        if torch.cuda.is_available():
            Train_dataset = torch.utils.data.TensorDataset(total_sample.cuda(), total_label.cuda())
            train_loader = torch.utils.data.DataLoader(Train_dataset, shuffle=True, batch_size = BATCH_SIZE)
            #Train_dataset = torch.utils.data.TensorDataset(sample_tensor.cuda(), label_tensor.cuda())
            #train_loader = torch.utils.data.DataLoader(Train_dataset, shuffle=True, batch_size = BATCH_SIZE)
        else:
            #Train_dataset = torch.utils.data.TensorDataset(total_sample, total_label)
            #train_loader = torch.utils.data.DataLoader(Train_dataset, shuffle=True, batch_size = BATCH_SIZE)
            Train_dataset = torch.utils.data.TensorDataset(sample_tensor, label_tensor)
            train_loader = torch.utils.data.DataLoader(Train_dataset, shuffle=True, batch_size = BATCH_SIZE)

        #### declare models
        net = Net(n_feature=input_dim, n_hidden=h1_dim, n_hidden2 = h2_dim, n_hidden3 = h3_dim, n_hidden4 = h4_dim, n_output=output_dim)     # define the network
        print(net)  # net architecture

        if torch.cuda.is_available():
            net.cuda()

        optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay = 0.0)
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
                    pred_y = prediction.cpu().data.numpy().squeeze()
                    target_y = y.cpu().data.numpy()
                    #plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
                    accuracy = sum(pred_y == target_y)/BATCH_SIZE
                    print(accuracy)
                    #plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
                    #plt.pause(0.1)
            net.eval()
            test_tensor1 = Variable(test_tensor)
            out = net(test_tensor1) 
            prediction = torch.max(F.softmax(out), 1)[1]
            pred_y = prediction.cpu().data.numpy().squeeze()
            target_y = test_label_tensor.cpu().numpy()
            accuracy = sum(pred_y == target_y)/target_y.shape[0]
            print(accuracy)



        net.eval()
        sample_tensor = Variable(sample_tensor)
        out = net(sample_tensor.cuda()) 
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.cpu().data.numpy().squeeze()
        target_y = label_tensor.cpu().squeeze().numpy()
        #print(pred_y)
        #print(target_y)
        accuracy = sum(pred_y == target_y)/target_y.shape[0]
        print(accuracy)

        test_tensor = Variable(test_tensor)
        out = net(test_tensor) 
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.cpu().data.numpy().squeeze()
        target_y = test_label_tensor.cpu().numpy()
        accuracy = sum(pred_y == target_y)/target_y.shape[0]
        print(accuracy)