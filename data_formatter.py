import numpy as np
import matplotlib.pyplot as plt
import ast

def evaluateValues(arr):
    vals = []
    for x in arr:
        if(float(ast.literal_eval(x)) > 0.0):
            vals += [1.0]
        else:
            vals += [float(ast.literal_eval(x))]

    return(np.array(vals))

def generateImage(str):
    arr = np.array(str.split(" "))
    image = np.reshape(evaluateValues(arr), (28, 28, 3))
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
    return (imgs)


if __name__ == "__main__":
    imgs = formatData()
    print(len(imgs))
    inp = input('enter Image number to view: ')
    while (inp != -1):
        try:
            plt.imshow(imgs[int(inp)])
            plt.show()
            inp = input('enter Image number to view: ')
        except:
            print("Invalid value")
