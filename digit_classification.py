import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        
        x = self.conv1(x)
        
        x = F.relu(x)
        
        x = self.conv2(x)
        
        x = F.relu(x)
        
        x = F.max_pool2d(x, 2)
        
        x = self.dropout1(x)
        
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        
        x = F.relu(x)
        
        x = self.dropout2(x)
        
        x = self.fc2(x)
        
        output = F.log_softmax(x, dim=1)
        return output

import torch.optim as optim
from torchvision import datasets, transforms
import wandb

# Initialize Weights & Biases
wandb.init(project='mnist-digit-recognition')

# Define the training and test sets
train_dataset = datasets.MNIST('../data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
test_dataset = datasets.MNIST('../data', train=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))

# Define the data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)

# Define the model, optimizer, and loss function
model = Net()
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Define the number of epochs to train for
num_epochs = 10

# Define the path to save the checkpoint
checkpoint_path = 'model_checkpoint.pt'

# Train the model
for epoch in range(1, num_epochs + 1):
    # Set the model to training mode
    model.train()

    # Train the model for one epoch
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Log the loss and training progress to wandb
        wandb.log({'Train Loss': loss.item(), 'Epoch': epoch, 'Batch': batch_idx})

    # Evaluate the model on the test set
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Print the results for this epoch
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('Epoch [{}/{}], Test Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(
        epoch, num_epochs, test_loss, accuracy))

    # Log the test loss and accuracy to wandb
    wandb.log({'Test Loss': test_loss, 'Test Accuracy': accuracy, 'Epoch': epoch})

    # Save a checkpoint of the model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)

    # Log the checkpoint file to wandb
    wandb.save(checkpoint_path)
