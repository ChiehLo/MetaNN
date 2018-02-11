import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as Data
import argparse

parser = argparse.ArgumentParser(description='PDX datasets')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--dropout', type=float, default=0.0, metavar='S',
                    help='dropout rate (default: 0)')
args = parser.parse_args()


RSEED = args.seed
drop_prob = args.dropout
torch.manual_seed(RSEED)    # reproducible
if torch.cuda.is_available():
    torch.cuda.set_device(0)

BATCH_SIZE = 32
EPOCH = args.epochs
LR = 0.005


input_dim = 5955
h1_dim = 512
h2_dim = 256
h3_dim = 128
output_dim = 4
CNN_flag = False

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_hidden2, n_hidden3, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden2)   # hidden layer
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)   # hidden layer
        self.dropout1 = torch.nn.Dropout(p=drop_prob)   # hidden layer
        self.out = torch.nn.Linear(n_hidden3, n_output)   # output layer
    def forward(self, x):
        x = self.hidden(x)
        x = self.dropout1(x)
        x = F.relu(x)      # activation function for hidden layer
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden3(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.out(x)
        return x

class CNN1d(torch.nn.Module):
    def __init__(self, n_feature, out_dim, n_output):
        super(CNN1d, self).__init__()
        self.c1 = torch.nn.Conv1d(1,8,3, stride = 2, padding = 1)
        self.c2 = torch.nn.Conv1d(8,8,3, stride = 2, padding = 1)
        self.p1 = torch.nn.MaxPool1d(2)   # hidden layer
        self.p2 = torch.nn.MaxPool1d(2)   # hidden layer
        self.dropout1 = torch.nn.Dropout(p=drop_prob)   # hidden layer
        self.out = torch.nn.Linear(out_dim, n_output)   # output layer
    def forward(self, x):
        x = F.relu(self.dropout1(self.c1(x)))      # activation function for hidden layer
        x = self.p1(x)
        x = F.relu(self.c2(x))
        x = self.p2(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x



IT = np.array([1, 2, 3, 4, 5])
pp = np.array([0, 50, 75, 100, 150, 200, 400, 800, 1200])


prefix = '../../Data/PDX/Splits/IT'
prefix_aug = '../../Data/PDX/Augmentation/IT'
if CNN_flag:
    prefix_save = '../../Data/PDX/Results_CNN/' + str(RSEED) + '/IT'
else:
    prefix_save = '../../Data/PDX/Results_MLP/' + str(RSEED) + '/IT'

for it in range(len(IT)):
    #A_D_200
    if drop_prob > 0:
        save_file = prefix_save + str(IT[it]) + '_A_D_' + str(EPOCH) + '_' + str(BATCH_SIZE)  + '.txt'
    else:
        save_file = prefix_save + str(IT[it]) + '_A_ND_' + str(EPOCH) + '_' + str(BATCH_SIZE)  + '.txt'
    m = open(save_file, "a")
    #### load training samples
    sample_file = prefix + str(IT[it]) + '/TrainSample_' + str(IT[it]) + '.txt'
    label_file = prefix + str(IT[it]) + '/TrainLabel_' + str(IT[it]) + '.txt'
    train_sample = np.loadtxt(sample_file, delimiter='\t')
    train_label = np.loadtxt(label_file, delimiter='\t')
    row_sums = train_sample.sum(axis=1)
    train_sample = train_sample / row_sums[:, np.newaxis]
    
    

    sample_tensor = torch.from_numpy(train_sample).type(torch.FloatTensor)
    label_tensor = torch.from_numpy(train_label).type(torch.LongTensor)

    #### load testing samples
    sample_file = prefix + str(IT[it]) + '/TestSample_' + str(IT[it]) + '.txt'
    label_file = prefix + str(IT[it]) + '/TestLabel_' + str(IT[it]) + '.txt'
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
    if CNN_flag:
        test_tensor = test_tensor.view(test_tensor.size(0), 1, test_tensor.size(1))
        sample_tensor = sample_tensor.view(sample_tensor.size(0), 1, sample_tensor.size(1))

    #### load nb data
    for cc in range(len(pp)):
        if pp[cc] == 0:
            total_sample = sample_tensor
            total_label = label_tensor
        else:
            sample_file = prefix_aug + str(IT[it]) + '/nbSample_' + str(IT[it]) + '_' + str(pp[cc]) + '.txt'
            label_file = prefix_aug + str(IT[it]) + '/nbLabel_' + str(IT[it]) + '_' + str(pp[cc]) + '.txt'
            train_sample = np.loadtxt(sample_file, delimiter='\t')
            train_label = np.loadtxt(label_file, delimiter='\t')
            row_sums = train_sample.sum(axis=1)
            train_sample = train_sample / row_sums[:, np.newaxis]
            vae_tensor = torch.from_numpy(train_sample).type(torch.FloatTensor)
            vae_label_tensor = torch.from_numpy(train_label).type(torch.LongTensor)
            if CNN_flag:
                vae_tensor = vae_tensor.view(vae_tensor.size(0), 1, vae_tensor.size(1))
            total_sample = torch.cat((sample_tensor, vae_tensor), 0)
            total_label = torch.cat((label_tensor, vae_label_tensor), 0)

        if torch.cuda.is_available():
            Train_dataset = torch.utils.data.TensorDataset(total_sample.cuda(), total_label.cuda())
            train_loader = torch.utils.data.DataLoader(Train_dataset, shuffle=True, batch_size = BATCH_SIZE)
        else:
            Train_dataset = torch.utils.data.TensorDataset(sample_tensor, label_tensor)
            train_loader = torch.utils.data.DataLoader(Train_dataset, shuffle=True, batch_size = BATCH_SIZE)


        #### declare models
        if CNN_flag:
            net = CNN1d(n_feature=input_dim, out_dim = 2976, n_output= output_dim)
        else:
            net = Net(n_feature=input_dim, n_hidden=h1_dim, n_hidden2 = h2_dim, n_hidden3 = h3_dim, n_output=output_dim)  

        if torch.cuda.is_available():
            net.cuda(0)


        optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay = 0.0)
        loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

        for epoch in range(EPOCH):
            net.train()
            for step, (X, Y) in enumerate(train_loader):
                x = Variable(X)   # batch x, shape (batch, 28*28)
                Y = torch.squeeze(Y)
                if torch.cuda.is_available():
                    Y = Y.cuda()
                y = Variable(Y)   # batch y, shape (batch, 28*28)

                out = net(x)                 # input x and predict based on x
                loss = loss_func(out, y)
        
                optimizer.zero_grad()               # clear gradients for this training step
                loss.backward()                     # backpropagation, compute gradients
                optimizer.step()                    # apply gradients

                if step % 20 == 0:
                    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0])
                    prediction = torch.max(F.softmax(out), 1)[1]
                    pred_y = prediction.cpu().data.numpy().squeeze()
                    target_y = y.cpu().data.numpy()
                    accuracy = sum(pred_y == target_y)/BATCH_SIZE

            net.eval()
            test_tensor1 = Variable(test_tensor)
            out = net(test_tensor1) 
            prediction = torch.max(F.softmax(out), 1)[1]
            pred_y = prediction.cpu().data.numpy().squeeze()
            target_y = test_label_tensor.cpu().numpy()
            accuracy = sum(pred_y == target_y)/target_y.shape[0]
            print(accuracy)




        net.eval()
        sample_tensor1 = Variable(sample_tensor)
        if torch.cuda.is_available():
            out = net(sample_tensor1.cuda()) 
        else:
            out = net(sample_tensor1)
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.cpu().data.numpy().squeeze()
        target_y = label_tensor.cpu().squeeze().numpy()
        accuracy = sum(pred_y == target_y)/target_y.shape[0]
        print(accuracy)

        test_tensor1 = Variable(test_tensor)
        out = net(test_tensor1) 
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.cpu().data.numpy().squeeze()
        target_y = test_label_tensor.cpu().numpy()
        accuracy = sum(pred_y == target_y)/target_y.shape[0]
        print(accuracy)
        probs = F.softmax(out).cpu().data.numpy()
        if CNN_flag:
            prefix_save_prob = '../../Data/PDX/Results_CNN/' + str(RSEED) + '/prob/IT'
        else:
            prefix_save_prob = '../../Data/PDX/Results_MLP/' + str(RSEED) + '/prob/IT'
        if drop_prob > 0:
            save_prob = prefix_save_prob + str(IT[it]) + '_A_D_Prob_' + str(EPOCH) + '_' + str(BATCH_SIZE) + '_' + str(pp[cc]) + '.txt'
        else:
            save_prob = prefix_save_prob + str(IT[it]) + '_A_ND_Prob_' + str(EPOCH) + '_' + str(BATCH_SIZE) + '_' + str(pp[cc]) + '.txt'
        np.savetxt(save_prob, probs, fmt='%1.4f', delimiter='\t')
        m.write(str(accuracy))
        m.write('\n')
    m.close()

