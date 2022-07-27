---
title: "Linear Regression in PyTorch"
date: 2020-02-18
toc: true
toc_label: "Contents"
toc_sticky: True
excerpt: "Learn how to do Linear Regression in PyTorch."
categories:
  - Programming
  - Machine Learning
tags:
  - Python
  - PyTorch
  - Regression
---

Hello and welcome to this tutorial where we learn a quadratic equation in PyTorch. We will train a neural network to predict future values of the function $y=2x$. So given an $x$, the network will have to predict $y$. 

This is similar to a previous tutorial that I made which also learns  a function but that was in Tensorflow.js. Check out that tutorial at this [link](https://brave-lewin-ba468b.netlify.com/2019-12-02-machine-learning-in-the-browser/).

Let's start coding!

> Before you start, I recommend that you open a Jupyter Notebook and run the code there. If you don't have Jupyter Notebook locally, you can either install it or use Google Colaboratory which is an online Jupyter Notebook editor.

## Importing Modules
We will import all necessary models in the following cell:

```python
import numpy as np # Numpy
import matplotlib.pyplot as plt # Matplotlib
import torch # PyTorch
```

1. We import numpy for working with our x and y data.
2. We import Matplotlib so that we can plot our function.
3. We import PyTorch in the 3rd line to build and train our network

## Generating the Data

Let's generate the data now that we have imported all the modules that we need, we can generate the x and y data. The following cell does that for us.

```python
x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values = [2*i for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)
```

The first 3 lines generate the x data which is just a list of numbers from 1 to 11 (11 is excluded). 

The last three lines generate the y data which is $x^2$.

We also have to reshape the data so that we can feed it to the neural network.

Let's plot the data.

```python
plt.scatter(x_train, y_train)
```

**Please Put Image Here**

We can imagine the curve going through the line and we want the neural network to generate that curve. Now that we generated the data, we can build and train a neural network.

## Building the network

We will now build the neural network. We are importing `nn` from PyTorch and we are importing `functional` which gives us the Leaky ReLU activation function.
```python
import torch.nn as nn
import torch.nn.functional as F
class Regression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LinearRegression, self).__init__()
        self.fc1 = nn.Linear(inputSize, 25)
        self.fc2 = nn.Linear(25, outputSize)

    def forward(self, x):
        out = F.leaky_relu(self.fc1(x))
        return self.fc2(out)
```

Our network has two layers. One layer is the input layer and the other is the output layer. A sequential model would have worked just fine here since this is very simple. Let's initialize our neural network so that we can train it.

```python
inputDim = 1
outputDim = 1
learningRate = 0.01 
epochs = 1500

model = Regression(inputDim, outputDim)
```

Our input and output dimensions are 1 since we are feeding in 1 number and returning 1 number. You can see that we have initialized our learning rate and that we are going to train for 1500 epochs. I recommend you train a bit more but 1500 will do.

Let's train the network now since we defined its architecture and initialized it.

## Training the Neural Network

Before we train the neural network, we have to define the loss function and the optimizer. The following cell does that for us.

```python
criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
```

We are using MSE for our loss since we have a regression problem. We are using Adam as our optimizer and our learning rate is set to the learning rate which was defined above, which is 0.01

We can now train the network since we have defined our loss function and our optimizer. The training loop is very simple. Here it is:

```python
for epoch in range(epochs):
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))
```

We print the loss every 100 epochs. If the model trains correctly, the loss should be decreasing and there should not be any errors. The output should a look a little like this:

```
epoch 0, loss 2550.587890625
epoch 100, loss 147.85638427734375
epoch 200, loss 73.8399429321289
epoch 300, loss 70.25251007080078
epoch 400, loss 65.93231964111328
epoch 500, loss 59.297264099121094
epoch 600, loss 51.42591094970703
epoch 700, loss 40.72663879394531
epoch 800, loss 30.04986000061035
epoch 900, loss 20.650432586669922
epoch 1000, loss 13.261860847473145
epoch 1100, loss 8.31400203704834
epoch 1200, loss 5.577629089355469
epoch 1300, loss 3.871095657348633
epoch 1400, loss 3.007876396179199
```

You can see that the loss decreases from 2550 to just 3. This shows that our model is improving and learning. That is a good sign that our model has learned the function. 

## Testing the model

To make sure that our model has learned the function, we can test our model. We do that in the following code:

```python
with torch.no_grad(): # we don't need gradients in the testing phase
    predicted = model(torch.from_numpy(x_train)).data.numpy()

plt.clf()
plt.plot(x_train, y_train, 'go', label='True data')
plt.plot(x_train, predicted, '--', label='Predictions')
plt.title("Linear Regression (y=2x)")
plt.legend(loc='best')
plt.show()
```

If it works, you should have a graph with a line that goes almost perfectly through the points.

## Conclusion

We have successfully built a model to do Linear Regression in PyTorch and we can see that it is very accurate. You can play around with the model and try to predict future values with it. That's it for now and I will see you later!

> If you liked this tutorial, check out my other tutorials and my other blog [here](https://brave-lewin-ba468b.netlify.com).