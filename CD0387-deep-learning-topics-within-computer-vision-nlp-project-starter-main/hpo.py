#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse

def test(model, test_loader, criterion):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            
            output = model(data)
            test_loss += criterion(output, target  # sum up batch loss
            outputs = model(data)          
            loss = criterion(outputs, target)

            running_loss += loss.item() * data.size(0)  # necessary, because CrossEntropyLoss returns the average loss over each element of a mini-batch

            _, preds = torch.max(outputs, 1)  # gets highest prediction (probability for a class)

            correct += torch.sum(preds == target).item()

    print("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset))


def train(model, train_loader, validation_loader, criterion, optimizer, epochs, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''

    for e in range(epoch):
        for phase in ['train','validation']
            running_loss=0
            correct=0

            if phase == 'train':
                mode.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = validation_loader

            for data, target in data_loader:
                data = data.to(device)
                target = target.to(device)

                outputs = model(data)          
                loss = criterion(outputs, target)
                
                running_loss += loss.item() * data.size(0)  # necessary, because CrossEntropyLoss returns the average loss over each element of a mini-batch

                _, preds = torch.max(outputs, 1)  # gets highest prediction (probability for a class)
                
                correct += torch.sum(preds == target).item()
                
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
        print(f"Epoch {e}: Loss {running_loss/len(train_loader.dataset)}, \
            Accuracy {100*(correct/len(train_loader.dataset))}%")

    
    return model
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    
    model = models.resnet18(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 133))  # 133 = classes for our dataset
    
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    
    loaded_data_dict = {}
    for split_dataset in data:
        data_loader = torch.utils.data.DataLoader(split_dataset, args.batch_size, shuffle=True)
        loaded_data_dict[split_dataset] = data_loader
        
    return loaded_data_dict
    
def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")

    model=model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    
    # add transforms for data-loaders
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    datasets = {}
    for datasplit in ['train','validation','test']:
        split_dataset = datasets.ImageFolder(os.path.join(args.data_dir, datasplit), data_transforms[datasplit])
        datasets[datasplit] = split_dataset
        
    # create data loaders
    dataloaders = create_data_loaders(datasets, args.batch_size)
    train_loader = dataloaders['train']
    validation_loader = dataloaders['valid']
    test_loader = dataloaders['test']
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model, train_loader, validation_loader, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion)
    
    '''
    TODO: Save the trained model
    '''
    path = os.path.join(args.model_dir, 'image_classification_model.pth')
    torch.save(model, path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    
    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    
    # container arguments
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    
    args=parser.parse_args()
    
    main(args)
