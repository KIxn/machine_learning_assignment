import torch


if __name__ == "__main__":
    model = torch.load('./digit_classifier_model.pt')
    print("model loaded")
    print('\n')
    model.eval()
    print(model)
    
    