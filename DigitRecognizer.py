import numpy as np
import matplotlib.pyplot as plt
import ast
import math
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

def batchGenerator(data , labels):
    for i in range(15):
        
        print(0*i)
        print(80*i)
        print('\n')
        
    


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
        Two_Dim = np.full((1,28, 28), 0)
        for r in range(28):

            for c in range(28):
                ans = 0
                
                ans += 0.2126* img[r][c][0]   #red
                ans += 0.7152* img[r][c][1]   #green
                ans += 0.0722* img[r][c][2]   #blue
                
                if (ans != 0):
                    ans = 1
                    
                    
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
    imgs = formatData() # (28, 28, 3)

    
    refinedImgs = refineData(imgs)
    img_data = torch.tensor(refinedImgs)
    labels = formatLabels()
    tl = torch.tensor(labels)
    
    
    training_data = img_data[0:1200]
    traning_labels = tl[0:1200]
    
    #trainloader = torch.utils.data.DataLoader(training_data,batch_size=64,shuffle=True)
    

    
    valid_data = img_data[1200:1600]
    valid_labels = tl[1200:1600]
    
    #valloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    
    test_data = img_data[1600:2000]
    test_labels = tl[1600:2000]
    
    #testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    
    def iterateData(data , lbl,i): # feed data in batches
        j = i - 1
        return data[j*80:i*80] , lbl[j*80:i*80]
    
    
    
        
        
        
    
    
    
    print(img_data.shape)
    print(training_data.shape)
    print(valid_data.shape)
    print(test_data.shape)
    
    
    print(tl.shape)
    print(len(refinedImgs[2]))
    inp = int(input('enter Image imgber to view: '))

    # while (inp != -1):
    #     try:
    #         plt.imshow(imgs[inp])
    #         plt.show()
    #         inp = int(input('enter Image imgber to view: '))
    #     except:
    #         print("Invalid value")

    while (inp != -1):
        try:
            print(np.array2string(imgs[inp]))
            print('\n')
            print(np.array2string(refinedImgs[inp]))
            print('\n')
            print("predicted num:")
            print(labels[inp])
            inp = int(input('enter Image number to view: '))
        except:
            print("Invalid value")