import torch
import torch.nn as nn
import torch.nn.functional as F



class CNNModel(nn.Module):
    def __init__(self,dropout):
        super(CNNModel,self).__init__()
       
        self.dropout = dropout

        self.conv1 = nn.Conv2d(1,10, kernel_size = 5) #1*5*5 input image,10*5*5 output channels, 5x5 square convolution
        self.conv2 = nn.Conv2d(10,20, kernel_size = 5) #10 output from conv1, 20 output channel, 5x5 square convolution
        self.dropout2d = nn.Dropout2d()
        self.linear1 = nn.Linear(320,50)
        self.linear2 = nn.Linear(50,10)
        
    def forward(self,x):

                          #Input: 100 x 1 x 28 x 28
        x = self.conv1(x) #Output1: 100 x 10 x 24 x 24
        x = F.max_pool2d(x, kernel_size = 2) #Output2 100 x 10 x 12 x 12
        x = F.relu(x)

        x = self.conv2(x) #Output3 100 x 20 x 8 x 8
        x = self.dropout2d(x)
        x = F.max_pool2d(x, kernel_size = 2) #Output4 100 x 20 x 4 x 4
        x = F.relu(x) 

        x = x.reshape(-1,320) #Input2 100 x 320
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout) #Output Linear1 100 x 50
        output = self.linear2(x) #Output Final 100 x 10
        output_softmax = F.log_softmax(output, dim = 1) #along the mathematical x-axis
        return output_softmax #is used when assigning an instant to one class when the number of possible classes is larger than two

def SaveModel(model,path = "Checkpoint.pth"):
    model.to("cpu")
    torch.save(model.state_dict(), path)
    print('Model has been saved')

#load model stat dict from:
def LoadModel(path = "Checkpoint.pth"):
    print('Model is Loading')
    model = CNNModel(0.5)
    model.load_state_dict(torch.load(path))
    print('Model has been loaded')
    return model





        
        


        

