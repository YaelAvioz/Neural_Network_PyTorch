import torch
from torch import nn

def write_res(results):
    with open('test_y', 'w') as out:
        for i in results:
            out.write("%s\n" % i)

def train(model, train_loader, optimizer):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.nll_loss(output, labels)
        loss.backward()
        optimizer.step()

#TODO: creat validate function
def validate():
    pass

def test(model, test_loader)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += nn.nll_loss(output, target, size_average = False).item()
            pred = output.max(1, keepdim = True)[1]
            correct += pred.eq(target.view_as(pred)).cpu().sum()

        # print the average loss
        test_loss /= len(test_loader.dataset)
        print('\nTest Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

#TODO: predict test_y on the test_x file

#TODO: write the result to the file

# Model A + Model B
class TwoLayersNetwork(nn.Module):
    def __init__(self, image_size):
        super(TwoLayersNetwork, self).__init__()

        # Neural Network with two hidden layers.
        # The first layer have a size of 100 and the second layer have a size of 50.
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    # function activation - ReLU
    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = nn.relu(self.fc0(x))
        x = nn.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.log_softmax(x)

# Model C
class TwoLayersNetworkDropout(nn.Module):
    def __init__(self, image_size):
        super(TwoLayersNetworkDropout, self).__init__()
        self.image_size = image_size

        # Same Neural Network as Model B:
        # Neural Network with two hidden layers.
        # The first layer have a size of 100 and the second layer have a size of 50.
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    # function activation - ReLU
    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = nn.relu(self.fc0(x))
        # Best Dropout rate (p=0.5)
        m = nn.Dropout(p=0.5)
        x = m(x)
        x = nn.relu(self.fc1(x))
        m = nn.Dropout(p=0.5)
        x = m(x)
        x = self.fc2(x)
        return nn.log_softmax(x)

# Model D
class TwoLayersNetworkBatchNormalization(nn.Module):
    def __init__(self, image_size):
        super(TwoLayersNetworkBatchNormalization, self).__init__()

        # Neural Network with two hidden layers.
        # The first layer have a size of 100 and the second layer have a size of 50.
        self.image_size = image_size

        # Batch Normalization before activation function.
        self.fc0 = nn.Linear(image_size, 100)
        self.fc0_bn = nn.BatchNorm1d(100)
        self.fc1 = nn.Linear(100, 50)
        self.fc1_bn = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 10)
        self.fc2_bn = nn.BatchNorm1d(10)

    # function activation - ReLU
    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = nn.relu(self.fc0(x))
        x = nn.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.log_softmax(x)

    #TODO: Batch Normalization after activation function.

# Model E
class FiveLayersNetworkReLU(nn.Module):
    def __init__(self, image_size):
        super(FiveLayersNetworkReLU, self).__init__()

        # Neural Network with five hidden layers: [128,64,10,10,10]
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    # function activation - ReLU
    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = nn.relu(self.fc0(x))
        x = nn.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.log_softmax(x)

# Model F
class FiveLayersNetworkSigmoid(nn.Module):
    def __init__(self, image_size):
        super(FiveLayersNetworkSigmoid, self).__init__()

        # Neural Network with five hidden layers: [128,64,10,10,10]
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    # function activation - sigmoid
    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = torch.sigmoid(self.fc0(x))
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return nn.log_softmax(x)

if __name__ == "__main__":

    #TODO: load the data

    epoches = 10

    # Model A
    lr = 0.1 # TODO: check which is the best value for model A learning rate
    model_A = TwoLayersNetwork(image_size = 28 * 28)
    optimizer = torch.optim.SGD(model_A.parameters(), lr = lr)

    for i in range(epoches):
        train(i, model_A, train_loader, lr, optimizer)
        validate()
        #TODO: test - check the success rate (delete before submission)
        test_predictions = test(model_A, test_loader)

    # Model B
    lr = 0.1 # TODO: check which is the best value for model B learning rate
    model_B = TwoLayersNetwork(image_size = 28 * 28)
    optimizer = torch.optim.Adam(model_B.parameters(), lr=lr)

    for i in range(epoches):
        train(i, model_B, train_loader, lr, optimizer)
        validate()
        # TODO: test - check the success rate (delete before submission)
        test_predictions = test(model_B, test_loader)

    # Model C
    lr = 0.1 # TODO: check which is the best value for model C learning rate
    model_C = TwoLayersNetworkDropout(image_size = 28 * 28)
    optimizer = torch.optim.Adam(model_C.parameters(), lr = lr)

    for i in range(epoches):
        train(i, model_C, train_loader, lr, optimizer)
        validate()
        # TODO: test - check the success rate (delete before submission)
        test_predictions = test(model_C, test_loader)

    # Model D
    lr = 0.1 # TODO: check which is the best value for model D learning rate
    model_D = TwoLayersNetworkBatchNormalization(image_size=28 * 28)
    optimizer = torch.optim.SGD(model_D.parameters(), lr=lr)

    for i in range(epoches):
        train(i, model_D, train_loader, lr, optimizer)
        validate()
        # TODO: test - check the success rate (delete before submission)
        test_predictions = test(model_D, test_loader)

    #TODO: for model E and F they didnt say which optimizer we should use so you need to check different
    # optimizers and check which one is the best - for now i wrote SGD

    # Model E
    lr = 0.1# TODO: check which is the best value for model D learning rate
    model_E = FiveLayersNetworkReLU(image_size = 28 * 28)
    #TODO: check what is the best optimizer for this model
    optimizer = torch.optim.SGD(model_E.parameters(), lr=lr)

    for i in range(epoches):
        train(i, model_E, train_loader, lr, optimizer)
        validate()
        # TODO: test - check the success rate (delete before submission)
        test_predictions = test(model_E, test_loader)

    # Model F
    lr = 0.1 # TODO: check which is the best value for model D learning rate
    model_F = FiveLayersNetworkSigmoid(image_size = 28 * 28)
    # TODO: check what is the best optimizer for this model
    optimizer = torch.optim.SGD(model_F.parameters(), lr=lr)

    for i in range(epoches):
        train(i, model_F, train_loader, lr, optimizer)
        validate()
        # TODO: test - check the success rate (delete before submission)
        test_predictions = test(model_F, test_loader)

    # TODO: after checks all the models choose the best one and write the result to test_y file
    # TODO: predict test_y on the test_x file
    write_res()
