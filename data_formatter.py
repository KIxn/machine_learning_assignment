import numpy as np
import matplotlib.pyplot as plt
import ast
import math
import torch
from torch import nn
from torch.utils.data import DataLoader
from time import time
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn, optim

def batchGenerator(data , labels):
    dataBatches = []
    labelBatches = []
    
    for i in range(15):
        dataBatches += [data[80*i:80*(i+1)]]
        labelBatches += [labels[80*i:80*(i+1)]]

    dataBatches = [torch.tensor(x) for x in dataBatches]
    labelBatches = [torch.tensor(x) for x in labelBatches]

    return(dataBatches,labelBatches)

#convert decimal rgb to rgb-255
def evaluateValues(arr):
    vals = []
    for x in arr:
        vals += [math.floor(ast.literal_eval(x) * 255.0)]

    return (np.array(vals))


def generateImage(str):
    arr = np.array(str.split(" "))
    image = np.reshape([evaluateValues(arr)], (28, 28, 3))
    return image


def formatData():
    imgs = []
    #take in data
    with open('inputs.txt', 'r') as f:
        while True:
            str = f.readline()
            if not str:
                break
            imgs += [generateImage(str)]

    f.close()
    return np.array(imgs)


#refine data to a 28x28 grid
def refineData(imgs):
    refinedImgs = []
    for img in imgs:
        Two_Dim = np.full((1,28, 28), 0.0)
        for r in range(28):

            for c in range(28):
                ans = 0
                
                ans += 0.2126* img[r][c][0]   #red
                ans += 0.7152* img[r][c][1]   #green
                ans += 0.0722* img[r][c][2]   #blue
                
                if (ans != 0):
                    ans = 1.0
                    
                    
                Two_Dim[0][r][c] = np.array(ans)
        refinedImgs += [Two_Dim]

    #return (np.reshape(np.array(refinedImgs), (28, 28, 1)))
    return (np.array(refinedImgs))


#take in labels
def formatLabels():
    imgs = []
    #take in data
    with open('labels.txt', 'r') as f:
        while True:
            str = f.readline()
            if not str:
                break
            imgs += [int(str)]

    f.close()
    return np.array(imgs)


if __name__ == "__main__":
    imgs = formatData()

    print(imgs[2].shape)
    refinedImgs = refineData(imgs)
    # t = torch.tensor(refinedImgs)
    labels = formatLabels()
    # tl = torch.tensor(labels)
    training_data,training_labels = batchGenerator(refinedImgs,labels)
    #print(type(t[12]))
    #print(t[12].shape) ([80, 1, 28, 28])
    
    
    #print(type(tl[12])) 
    #print(tl[12].shape)
    #print(len(refinedImgs[2]))
    
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]), nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
    print(model)

    criterion = nn.NLLLoss()
    images, labels = training_data[0] , training_labels[0]
    images = images.view(images.shape[0], -1)
    #print(images)
    logps = model(images.float())  #log probabilities
    loss = criterion(logps, labels.long())  #calculate the NLL loss

    print('Before backward pass: \n', model[0].weight.grad)
    loss.backward()
    print('After backward pass: \n', model[0].weight.grad)

    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    time0 = time()
    epochs = 15
    for e in range(epochs):
        running_loss = 0
        for b in range(15):
            images, labels = training_data[b] , training_labels[b]
            
                
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)

            # Training pass
            optimizer.zero_grad()

            output = model(images.float())
            loss = criterion(output, labels.long())

            #This is where the model learns by backpropagating
            loss.backward()

            #And optimizes its weights here
            optimizer.step()

            running_loss += loss.item()
            
            print("Epoch {} - Training loss: {}".format(e, running_loss / len(training_data)))
    print("\nTraining Time (in minutes) =", (time() - time0) / 60)
    
    torch.save(model, './digit_classifier_model.pt')
    print("model saved")

