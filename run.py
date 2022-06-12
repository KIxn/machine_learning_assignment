import torch
import numpy as np
import math
import ast
import sys


#convert decimal rgb to rgb-255
def evaluateValues(arr):
    vals = []
    for x in arr:
        vals += [(np.float(x) * 255.0)]

    return (np.array(vals))


if __name__ == "__main__":
    model = torch.load('./digit_classifier_model.pt')
    print("model loaded")
    print('\n')
    model.eval()
    print(model)
    arr = np.loadtxt(sys.stdin).reshape(-1, 28, 28, 3)
    labels = [int(x) for x in np.loadtxt('labels.txt')]
    correct = 0
    total = 0
    for inp in arr:
        img = inp  # np.reshape([evaluateValues(inp)], (28, 28, 3))
        Two_Dim = np.full((1, 28, 28), 0.0)
        for r in range(28):

            for c in range(28):
                ans = 0

                ans += 0.2126 * img[r][c][0]  #red
                ans += 0.7152 * img[r][c][1]  #green
                ans += 0.0722 * img[r][c][2]  #blue

                if (ans != 0):
                    ans = 1.0

                Two_Dim[0][r][c] = np.array(ans)
        image = torch.tensor(Two_Dim)
        img = image.view(1, 784)
        with torch.no_grad():
            logps = model(img.float())

        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred = probab.index(max(probab))
        print("Predicted: " + str(pred) + " Expected: " + str(labels[total]))
        if (pred == labels[total]):
            correct += 1
        total += 1

    accuracy = correct / total
    accuracy *= 100
    print('Accuracy: ' + str(accuracy) + '%')
