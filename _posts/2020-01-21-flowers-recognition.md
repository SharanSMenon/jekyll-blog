---
title: "Flower Recognition in PyTorch"
date: 2020-01-21
toc: true
toc_label: "Contents"
toc_sticky: True
excerpt: "Use transfer learning to classify images of flowers in PyTorch."
categories:
  - Programming
  - Machine Learning
tags:
  - Python
  - Computer Vision
  - PyTorch
---
Welcome to this tutorial where you will learn to use transfer learning to classify images of flowers with pytorch. We will be using the VGG16 model in the tutorial so follow along.

## Getting the Data

Here is the [link](https://www.kaggle.com/alxmamaev/flowers-recognition) to the dataset.

If you have a folder called `flowers` in your data folder, run this command or just remove it.
```python
!rm -r flowers/flowers
# This happens sometimes. 
# It can lead to problems later on
```

## Install PyTorch Ignite

> PyTorch Ignite helps you train your neural networks quickly and easily. It is a really useful tool and can speed up your ml workflow a lot.

If you do not have PyTorch Ignite Installed, run the following command

```
pip install pytorch-ignite
```

For conda users...

```
conda install 
```

## Importing Modules and Loading Data

We will import the modules needed in the next 4 cells
```python
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
```

Matplotlib for image viewing and plotting. We will use seaborn to make a heatmap of the confusion matrix later on.

```python
import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline
%config InlineBackend.figure_format = 'retina'
```

We import `models` from torchvision because we will be using a pretrained model.

```python
import numpy as np
from torchvision import models
```

We import PyTorch Ignite here

```python
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix
```

We will define our dataloaders here. We need to resize the images to 224 X 224 because that is what the model we are going to use requires. We also split the data into training, validation, and testing. 

The training and validation data will be used during the training of the model while the testing data will be used to see the performance of the model on unseen images.

```python
def get_data_loaders(data_dir, batch_size):
  transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
  all_data = datasets.ImageFolder(data_dir, transform=transform)
  train_data_len = int(len(all_data)*0.75)
  valid_data_len = int((len(all_data) - train_data_len)/2)
  test_data_len = int(len(all_data) - train_data_len - valid_data_len)
  train_data, val_data, test_data = random_split(all_data, [train_data_len, valid_data_len, test_data_len])
  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
  return ((train_loader, val_loader, test_loader), all_data.classes)
```

We get the three data loaders as well as the class names which you will see in the next cell

```python
(train_loader, val_loader, test_loader), classes = get_data_loaders("/content/flowers", 64)
```

Let's print the names of the classes here.

```python
classes
```



    (Out)
    ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


We have 5 classes here, `daisy`, `dandelion`, `rose`, `sunflower`, and `tulip`


We also print the length of the dataloaders.
```python
print(len(train_loader))
print(len(val_loader))
print(len(test_loader))
```

    51
    9
    9


We will display a batch of images here. We use `plt.imshow` to display the images.
```python
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
    ax.set_title(classes[labels[idx]])
```

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/2020-01-21/output_12_0.png" alt="">


## Building and Training the Model

If there is a GPU available, we should use it as a GPU will make training way quicker.

```python
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device
```


    (Out)
    'cuda:0'

I have a GPU available which is why it says `cuda:0`. If you do not have a GPU then it will say `cpu`.


We will be using a pretrained model. It is VGG16.
```python
model = models.vgg16(pretrained=True)
```

The model is really large so I cut a lot of it out and replaced it with `...`. The `classifier` part is the most important to us.

```python
print(model)
```
    (Out)
    VGG(
      (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ...
        (29): ReLU(inplace=True)
        (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
      (classifier): Sequential(
        (0): Linear(in_features=25088, out_features=4096, bias=True)
        (1): ReLU(inplace=True)
        (2): Dropout(p=0.5, inplace=False)
        (3): Linear(in_features=4096, out_features=4096, bias=True)
        (4): ReLU(inplace=True)
        (5): Dropout(p=0.5, inplace=False)
        (6): Linear(in_features=4096, out_features=1000, bias=True)
      )
    )


Let's see the amount of output classes the model returns.

```python
print(model.classifier[6].in_features) 
print(model.classifier[6].out_features)
```
    (Out)
    4096
    1000

This model has 1000 output classes while we only want 5 output classes. So we have to change that.

We don't want to train any of the parameters except for the last ones.

```python
for param in model.features.parameters():
    param.requires_grad = False
```

We are changing the last layer of the nn in order for our purposes.

```python
n_inputs = model.classifier[6].in_features
last_layer = nn.Linear(n_inputs, len(classes))
model.classifier[6] = last_layer
if torch.cuda.is_available():
    model.cuda()
print(model.classifier[6].out_features)
```
    (Out)
    5

We define the optimizer and the loss function here.

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=0.001)
```

We are recording the training and validation history in the following cell.

```python
training_history = {'accuracy':[],'loss':[]}
validation_history = {'accuracy':[],'loss':[]}
```

We write some code to dictate what happens during training. After training, a heatmap will be printed which is the confusion matrix. We can do this thanks to Ignite. This is why I imported Ignite earlier

```python
trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
evaluator = create_supervised_evaluator(model,
                                        device=device,
                                        metrics={
                                            'accuracy': Accuracy(),
                                            'loss': Loss(criterion),
                                            'cm':ConfusionMatrix(len(classes))
                                            })
@trainer.on(Events.ITERATION_COMPLETED)
def log_a_dot(engine):
    print(".",end="")

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    accuracy = metrics['accuracy']*100
    loss = metrics['loss']
    training_history['accuracy'].append(accuracy)
    training_history['loss'].append(loss)
    print()
    print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(trainer.state.epoch, accuracy, loss))
    
@trainer.on(Events.EPOCH_COMPLETED)   
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    accuracy = metrics['accuracy']*100
    loss = metrics['loss']
    validation_history['accuracy'].append(accuracy)
    validation_history['loss'].append(loss)
    print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(trainer.state.epoch, accuracy, loss))

@trainer.on(Events.COMPLETED)
def log_confusion_matrix(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    cm = metrics['cm']
    cm = cm.numpy()
    cm = cm.astype(int)
    fig, ax = plt.subplots(figsize=(10,10))  
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax,fmt="d")
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels') 
    ax.set_title('Confusion Matrix') 
    ax.xaxis.set_ticklabels(classes,rotation=90)
    ax.yaxis.set_ticklabels(classes,rotation=0)
```

We train the model in the following cell. 

```python
trainer.run(train_loader, max_epochs=4)
```
    (Out)
    ...................................................
    Training Results - Epoch: 1  Avg accuracy: 69.49 Avg loss: 1.12
    Validation Results - Epoch: 1  Avg accuracy: 68.15 Avg loss: 1.14
    ...................................................
    Training Results - Epoch: 2  Avg accuracy: 74.55 Avg loss: 0.90
    Validation Results - Epoch: 2  Avg accuracy: 74.07 Avg loss: 0.93
    ...................................................
    Training Results - Epoch: 3  Avg accuracy: 77.79 Avg loss: 0.78
    Validation Results - Epoch: 3  Avg accuracy: 77.41 Avg loss: 0.81
    ...................................................
    Training Results - Epoch: 4  Avg accuracy: 79.55 Avg loss: 0.71
    Validation Results - Epoch: 4  Avg accuracy: 78.89 Avg loss: 0.74


<img src="{{ site.url }}{{ site.baseurl }}/assets/images/2020-01-21/output_23_2.png" alt="">

## Testing the Model

Let's test the model. Here we are seeing the performance of the model for each class.

```python
test_loss = 0.0
class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))

model.eval() # eval mode

# iterate over test data
for data, target in test_loader:
    # move tensors to GPU if CUDA is available
    if torch.cuda.is_available():
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update  test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    if len(target) == 64:
      for i in range(64):
          label = target.data[i]
          class_correct[label] += correct[i].item()
          class_total[label] += 1

# calculate avg test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(len(classes)):
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
    (Out)
    Test Loss: 0.688975
    
    Test Accuracy of daisy: 72% (61/84)
    Test Accuracy of dandelion: 86% (120/138)
    Test Accuracy of  rose: 77% (68/88)
    Test Accuracy of sunflower: 77% (65/84)
    Test Accuracy of tulip: 81% (96/118)
    
    Test Accuracy (Overall): 80% (410/512)

Our model does pretty well on the test set with an accuracy of 80%. Train for a few more epochs and this will increase to over 90%. It does very good on dandelion and not so good with daisy.

```python
dataiter = iter(test_loader)
images, labels = dataiter.next()
images.numpy()

# move model inputs to cuda, if GPU available
if torch.cuda.is_available():
    images = images.cuda()

# get sample outputs
output = model(images)
# convert output probabilities to predicted class
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(preds_tensor.cpu().numpy())

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx].cpu(), (1, 2, 0)))
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                 color=("green" if preds[idx]==labels[idx].item() else "red"))
```

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/2020-01-21/output_26_0.png" alt="">

We can see that it got a majority of the images correctly classified. That's it for this tutorial and I hope you have fun with your neural network. Bye!