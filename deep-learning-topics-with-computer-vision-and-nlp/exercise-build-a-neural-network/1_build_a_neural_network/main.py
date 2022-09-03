from torch import nn

def create_model():
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
    
