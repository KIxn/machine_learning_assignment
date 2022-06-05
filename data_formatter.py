import numpy as np
import matplotlib.pyplot as plt
import ast
import math

#convert decimal rgb to rgb-255
def evaluateValues(arr):
    vals = []
    for x in arr:
        vals += [math.floor(ast.literal_eval(x) * 255.0)]

    return(np.array(vals))

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
    return (imgs)

#refine data to a 28x28 grid
def refineData(imgs):
    refinedImgs = []
    for img in imgs:
        Two_Dim=np.full((28, 28), 0)
        for r in range(28):

            for c in range(28):
                ans=0

                for i in range(3):
                    ans=ans+ math.pow(img[r][c][i],2)
                ans=ans/3
                Two_Dim[r][c]= math.floor(math.sqrt(ans))
        refinedImgs += [Two_Dim]

    return(refinedImgs)

if __name__ == "__main__":
    imgs = formatData()
    refinedImgs = refineData(imgs)
    print(len(imgs))
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
            inp = int(input('enter Image imgber to view: '))
        except:
            print("Invalid value")