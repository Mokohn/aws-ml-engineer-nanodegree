#TODO: Import packages you need
from torchvision import transforms, datasets
import torch
from torch import nn, optim

def train(model, train_loader, cost, optimizer, epoch):
    model.train()
    #TODO: Add your code here to train your model
    for e in range(epoch):
        running_loss = 0
        correct = 0  # for calculating accuracy
        for train, target in train_loader:
            data = train.view(train.shape[0], -1)  # The PyTorch view() function is used to convert 
                                                   # the tensor into a 2D format which is rows and columns. And we should have a definite number of rows and columns to view.
                                                   # here, -1 tells the function that it is an unknown dimension and that it figures it out by itself
            optimizer.zero_grad() # zero out gradients for batch
            '''
            In PyTorch, for every mini-batch during the training phase, we typically want to explicitly set the gradients to zero before starting to do backpropragation 
            (i.e., updating the Weights and biases) because PyTorch accumulates the gradients on subsequent backward passes. 
            This accumulating behaviour is convenient while training RNNs or when we want to compute the gradient of the loss summed over multiple mini-batches.
            '''
            pred = model(data)  # get predictions
            loss = cost(pred, target)  # calculate loss
            running_loss += loss  # add current loss to overall loss
            loss.backward()  # calculate gradients for model parameters
            optimizer.step()  # update model weights
            pred = pred.argmax(dim=1, keepdim=True)  # gets indices of maximum values in given tensor, here predictions, dim = dimension to reduce (here each list in given tensor), 
                                                     # keepdim = output has same dimension
            correct += pred.eq(target.view_as(pred)).sum().item()  # eq computed elementwise equality of given inputs, view_as = views tensor as same size as pred, 
                                                                   # item = Returns the value of this tensor as a standard Python number.
        print(f"EPoch {e}: Loss {running_loss/len(train_loader.dataset)}, Accuracy {100*(correct/len(train_loader.dataset))}%")
            
            

def test(model, test_loader):
    model.eval()
    #TODO: Add code here to test the accuracy of your model
    correct = 0
    with torch.no_grad():  # Disabling gradient calculation is useful for inference, when you are sure that you will not call Tensor.backward(). 
                           # It will reduce memory consumption for computations that would otherwise have requires_grad=True.
        for test, target in test_loader:
            data = test.view(test.shape[0], -1)
            output = model(data)
            pred   = output.argmax(dim=1, keepdim=True)  # get index of max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        print(f"Test-accuracy: {correct}/{len(test_loader.dataset)} = {100*(correct/len(test_loader.dataset))}")
    

def create_model():
    #TODO: Add your model code here. You can use code from previous exercises
    input_size = 784
    output_size = 10
    
    model = nn.Sequential(nn.Linear(input_size, 256),  # input layer
                         nn.ReLU(),
                         nn.Linear(256, 128),  # hidden layer
                         nn.ReLU(),
                          nn.Linear(128, 64),  # hidden layer
                          nn.ReLU(),
                          nn.Linear(64, output_size),  # output layer
                          nn.LogSoftmax(dim=1)
                         )
    return model

#TODO: Create your Data Transforms
# training transform with augmentation
training_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # data augmentation
    transforms.ToTensor(),  # transforms image to range of 0-1
    transforms.Normalize((0.1307,), (0.3081,))  # normalizes image
])

# testing transform without augmentation
testing_transform = transforms.Compose([
    transforms.ToTensor(),  # transforms image to range of 0-1
    transforms.Normalize((0.1307,), (0.3081,))  # normalizes image
])

# hyperparameters
batch_size = 128
epochs     = 5

#TODO: Download and create loaders for your data
trainset = datasets.MNIST('data/', download=True, train=True, transform=training_transform)
testset  = datasets.MNIST('data/', download=True, train=False, transform=testing_transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

model=create_model()

cost = nn.CrossEntropyLoss()  # crossentropy loss
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # adam optimizer

train(model, train_loader, cost, optimizer, epochs)
test(model, test_loader)
