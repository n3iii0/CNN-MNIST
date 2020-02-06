import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import random

from cnn_model import CNNModel, SaveModel, LoadModel


num_classes = 10
batch = 2000
learning_rate = 0.001

def plot(data_img, data_title):
        data_img = data_img.reshape(data_img.shape[1],data_img.shape[2])
        fig = plt.figure()
        plt.tight_layout()
        plt.imshow(data_img, cmap = 'gray', interpolation = 'none')
        plt.title(data_title)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(data_title)
        fig

def testing(model, data_set):
        i = random.randint(0,1001)
        test_inp, test_label = data_set[i]
        plot(test_inp, "result")
        test_inp = test_inp.unsqueeze(0)
        model.to('cuda')
        test_inp = test_inp.to('cuda')
        output = model.forward(test_inp)
        probabilities, results = output.topk(3)
        result = results.to('cpu').numpy()[0][0]
        print("The true value is {}, the neural network suggests {}".format(test_label, result))


data_path = 'C://Users//nedim//Desktop//cnn-test//Dataset'
data_transform = transforms.Compose([transforms.RandomRotation(90), transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
mnist_train_dataset = torchvision.datasets.MNIST(data_path, download = True, train = True, transform = data_transform)
mnist_valid_dataset = torchvision.datasets.MNIST(data_path, download = True, train = False, transform = data_transform)

train_loader = DataLoader(mnist_train_dataset, batch_size = batch, shuffle = True)
valid_loader = DataLoader(mnist_valid_dataset, batch_size = batch, shuffle = False)


model = CNNModel(0.5)
model.to('cuda')
epochs = 100
criterion = nn.NLLLoss()
counter = 0

valid_timing = 10
loss_graph = []
testloss_graph = []

optimizer = optim.Adam(model.parameters(), lr=learning_rate,)

'''
for e in range(epochs):
        print("{}/{}".format(e,epochs))
        for inp, label in train_loader:
                counter += 1
                optimizer.zero_grad()
                inp = inp.to('cuda')
                label = label.to('cuda')
                model.train()
                output = model.forward(inp)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                
                if counter % valid_timing == 0:
                    model.eval()
                    with torch.no_grad():
                        for valid_inp, valid_label in valid_loader:
                                valid_inp, valid_label = valid_inp.to('cuda'), valid_label.to('cuda')
                                output_valid = model.forward(valid_inp)
                                test_loss = criterion(output_valid, valid_label)
                                print("{}/{}".format(e,epochs))
                                print(loss.cpu().data.numpy())
                                loss_graph.append(loss.detach().to("cpu").numpy())
                                print(test_loss.cpu().data.numpy())
                                testloss_graph.append(test_loss.detach().to("cpu").numpy())
                                model.train()
SaveModel(model)
'''
LoadModel()

testing(model, mnist_valid_dataset)




