---
title: "CNNs in PyTorch"
date: 2020-01-05
toc: true
toc_label: "Contents"
toc_sticky: True
excerpt: "Train a CNN in Pytorch on the CIFAR-10 Dataset!"
categories:
  - Programming
  - Machine Learning
tags:
  - Python
  - Computer Vision
  - PyTorch
---

## Introduction

PyTorch is a deep learning library developed by Facebook. It is a powerful library and can build highly accurate deep learning models. We will build a Convolutional Neural Network today.

We will be training our model on the CIFAR-10 dataset which is a dataset of many 32 x 32 color images. Since there are a lot of images, you may want to use a GPU. 

## Test for [CUDA](http://pytorch.org/docs/stable/cuda.html)

Since these are larger (32x32x3) images, it may prove useful to speed up your training time by using a GPU. CUDA is a parallel computing platform and CUDA Tensors are the same as typical Tensors, only they utilize GPU's for computation. GPUs are better than CPUs for machine learning.


```python
import torch
import numpy as np

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
```

    CUDA is not available.  Training on CPU ...


---
## Load the [Data](http://pytorch.org/docs/stable/torchvision/datasets.html) and visualize it

Downloading may take a minute. We load in the training and test data, split the training data into a training and validation set, then create DataLoaders for each of these sets of data. We also augment the images so that the model gets better accuracy. Image Augmentation is a technique to avoid overfitting


```python
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20
# percentage of training set to use as validation
valid_size = 0.2

# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # randomly flip and rotate
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# choose the training and test datasets
train_data = datasets.CIFAR10('data', train=True,
                              download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=transform)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
    sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
    num_workers=num_workers)

# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
```
(Output)
    Files already downloaded and verified
    Files already downloaded and verified


### Visualize a Batch of Training Data

We show a small batch of training data here


```python
import matplotlib.pyplot as plt
%matplotlib inline

def imshow(img):
    img = img / 2 + 0.5
    plt.imshow(np.transpose(img, (1, 2, 0)))
```


```python
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
# display 25 images
for idx in np.arange(25):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])
```

### View an Image in More Detail

We look at the image in more detail by looking at the values for the 3 color channels. Remember that a computer sees a color image a 3 dimesional tensor (the height, width, and depth). It is basically 3 matrices (red, green, and blue)


```python
rgb_img = np.squeeze(images[3])
channels = ['red channel', 'green channel', 'blue channel']

fig = plt.figure(figsize = (36, 36)) 
for idx in np.arange(rgb_img.shape[0]):
    ax = fig.add_subplot(1, 3, idx + 1)
    img = rgb_img[idx]
    ax.imshow(img, cmap='gray')
    ax.set_title(channels[idx])
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            val = round(img[x][y],2) if img[x][y] !=0 else 0
            ax.annotate(str(val), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center', size=8,
                    color='white' if img[x][y]<thresh else 'black')
```

---
## Define the Network [Architecture](http://pytorch.org/docs/stable/nn.html)

We are building the architecture of the neural network in the following cell. We define 3 convolutional layers and a max pooling layers. We initialize the layers in the `__init__` function and use them in the `forward` function. We also define the loss function and optimizer in this section


```python
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer (sees 32x32x3 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # convolutional layer (sees 16x16x16 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # convolutional layer (sees 8x8x32 tensor)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (64 * 4 * 4 -> 500)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(500, 10)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x): # The Forward pass
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(-1, 64 * 4 * 4)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x
```

### Initialize an instance of the model


```python
model = Net()
print(model)

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()
```

### [Loss Function](http://pytorch.org/docs/stable/nn.html#loss-functions) and [Optimizer](http://pytorch.org/docs/stable/optim.html)

Here we define the loss function and the optimizer which will us improve our model as we train it


```python
import torch.optim as optim

# categorical cross-entropy
criterion = nn.CrossEntropyLoss()

# SGD Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

---
## Train the Network

We are training the network in the following cell. Every time the validation loss is lower than the currently lowest validation loss after each epoch, the model gets saved so that way the model with the lowest validation loss is loaded.

> If you want a final model, you should train somewhere between 30 and 50 epochs


```python
# number of epochs to train the model
n_epochs = 20

valid_loss_min = np.Inf # track change in validation loss

for epoch in range(1, n_epochs+1):

    # train and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    ###################
    # train the model #
    ###################
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
        
    ######################    
    # validate the model #
    ######################
    model.eval()
    for batch_idx, (data, target) in enumerate(valid_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
    
    # calculate average losses
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_augmented.pt')
        valid_loss_min = valid_loss
```

    Epoch: 1 	Training Loss: 2.034757 	Validation Loss: 1.778204
    Validation loss decreased (inf --> 1.778204).  Saving model ...
    Epoch: 2 	Training Loss: 1.670165 	Validation Loss: 1.592818
    Validation loss decreased (1.778204 --> 1.592818).  Saving model ...
    Epoch: 3 	Training Loss: 1.501479 	Validation Loss: 1.437829
    Validation loss decreased (1.592818 --> 1.437829).  Saving model ...
    Epoch: 4 	Training Loss: 1.407577 	Validation Loss: 1.332701
    Validation loss decreased (1.437829 --> 1.332701).  Saving model ...
    Epoch: 5 	Training Loss: 1.328776 	Validation Loss: 1.246034
    Validation loss decreased (1.332701 --> 1.246034).  Saving model ...
    Epoch: 6 	Training Loss: 1.259839 	Validation Loss: 1.190466
    Validation loss decreased (1.246034 --> 1.190466).  Saving model ...
    Epoch: 7 	Training Loss: 1.198763 	Validation Loss: 1.146982
    Validation loss decreased (1.190466 --> 1.146982).  Saving model ...
    Epoch: 8 	Training Loss: 1.147863 	Validation Loss: 1.083346
    Validation loss decreased (1.146982 --> 1.083346).  Saving model ...
    Epoch: 9 	Training Loss: 1.101410 	Validation Loss: 1.032714
    Validation loss decreased (1.083346 --> 1.032714).  Saving model ...
    Epoch: 10 	Training Loss: 1.059786 	Validation Loss: 0.985375
    Validation loss decreased (1.032714 --> 0.985375).  Saving model ...
    Epoch: 11 	Training Loss: 1.022977 	Validation Loss: 0.952992
    Validation loss decreased (0.985375 --> 0.952992).  Saving model ...
    Epoch: 12 	Training Loss: 0.993342 	Validation Loss: 0.921386
    Validation loss decreased (0.952992 --> 0.921386).  Saving model ...
    Epoch: 13 	Training Loss: 0.957348 	Validation Loss: 0.904237
    Validation loss decreased (0.921386 --> 0.904237).  Saving model ...
    Epoch: 14 	Training Loss: 0.927471 	Validation Loss: 0.865811
    Validation loss decreased (0.904237 --> 0.865811).  Saving model ...
    Epoch: 15 	Training Loss: 0.898602 	Validation Loss: 0.864282
    Validation loss decreased (0.865811 --> 0.864282).  Saving model ...
    Epoch: 16 	Training Loss: 0.882884 	Validation Loss: 0.836490
    Validation loss decreased (0.864282 --> 0.836490).  Saving model ...
    Epoch: 17 	Training Loss: 0.863756 	Validation Loss: 0.846190
    Epoch: 18 	Training Loss: 0.833526 	Validation Loss: 0.796503
    Validation loss decreased (0.836490 --> 0.796503).  Saving model ...
    Epoch: 19 	Training Loss: 0.821181 	Validation Loss: 0.788461
    Validation loss decreased (0.796503 --> 0.788461).  Saving model ...
    Epoch: 20 	Training Loss: 0.807184 	Validation Loss: 0.800496


###  Load the Model with the Lowest Validation Loss


```python
model.load_state_dict(torch.load('model_augmented.pt'))
```




    <All keys matched successfully>



---
## Testing the network
We are testing the network here and seeing its accuracy on the data. It achives around a 72% accuracy if you train it for 20 epochs. That is pretty good but we can improve the model a lot. Play around with the model and some other parameters to see if you can improve performance


```python
# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()
# iterate over test data
for batch_idx, (data, target) in enumerate(test_loader):
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
```

    Test Loss: 0.792076
    
    Test Accuracy of airplane: 74% (742/1000)
    Test Accuracy of automobile: 76% (769/1000)
    Test Accuracy of  bird: 59% (591/1000)
    Test Accuracy of   cat: 54% (541/1000)
    Test Accuracy of  deer: 66% (660/1000)
    Test Accuracy of   dog: 61% (619/1000)
    Test Accuracy of  frog: 84% (842/1000)
    Test Accuracy of horse: 75% (759/1000)
    Test Accuracy of  ship: 87% (871/1000)
    Test Accuracy of truck: 82% (827/1000)
    
    Test Accuracy (Overall): 72% (7221/10000)


### Test the model on a few sample images

We are testing the model on a few sample images


```python
# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()
images.numpy()

# move model inputs to cuda, if GPU available
if train_on_gpu:
    images = images.cuda()

# get sample outputs
output = model(images)
# convert output probabilities to predicted class
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                 color=("green" if preds[idx]==labels[idx].item() else "red"))
```

## Conclusion

We have build a CNN in PyTorch that can accurately classify the CIFAR-10 Dataset. I hope you enjoyed this tutorial and why don't you try building some CNNs on your own to classify other datasets
