---
title: "Train MNIST using only numpy"
date: 2021-08-18
toc: true
toc_label: "Contents"
toc_sticky: True
published: true
excerpt: "Create a neural network from scratch and train it on MNIST"
categories:
  - Programming
  - Machine Learning
tags:
  - Python
  - numpy
---

In this tutorial, we will use just numpy to create a neural network that gets 96% accuracy on the MNIST dataset. We won’t be using any DL Frameworks like Tensorflow or PyTorch.

> All training here is done on a MacBook Air M1

## Imports

```python
import numpy as np
import tensorflow as tf # Only for data
import matplotlib.pyplot as plt
```

We use tf.keras to load the data. We will not use tensorflow after that.

## Utility Functions

These utility functions will be used later

- `to_categorical` will be used for data preprocessing
- `accuracy_score` will evaluate the accuracy of the model
- `batch_iterator` splits the data into batches, so that way we dont have to pass the full dataset into the model and overload our computer.

```python
def to_categorical(x, n_col=None):
    """ One-hot encoding of nominal values """
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot
```

```python
def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy
```

```python
def batch_iterator(X, y=None, batch_size=64):
    """ Simple batch generator """
    n_samples = X.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i+batch_size, n_samples)
        if y is not None:
            yield X[begin:end], y[begin:end]
        else:
            yield X[begin:end]
```



## Data

The first line in the following segment is the only time we call tensorflow. The tensorflow function returns data as a numpy array so this makes things easy for us.

1. Preprocess the data
2. Reshape the data so we can input it into our model
3. Initialize the input_dim and the output_dim as info for our model.

```python
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
y_train, y_test = to_categorical(y_train.astype("int")), to_categorical(y_test.astype("int"))
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train, X_test = X_train.reshape(-1, 28 * 28), X_test.reshape(-1, 28 * 28)
# Now the X shape is (60000, 784), perfect for our linear model
# Y shape is (60000, 10), which is easy for us to train with
```

```python
n_input_dim = 784 # The input dimension for the model
n_out = 10 # 10 classes
```

## Loss and Activation

We define the loss and activation functions here. All 3 classes calculate the loss/activation and the gradient of that function.

This is our loss function.

```python
class CrossEntropy():
    def __init__(self): pass
    
    def loss(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def gradient(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)
```

`LeakyRELU()` is our input and hidden layers activation function.

```python
class LeakyReLU():
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, x):
        return self.activation(x)

    def activation(self, x):
        return np.where(x >= 0, x, self.alpha * x)

    def gradient(self, x):
        return np.where(x >= 0, 1, self.alpha)
```

Softmax is our output layer’s activation function

```python
class Softmax():
    def activation(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)
    
    def __call__(self, x):
        return self.activation(x)
```



## Layers

### Activation Layer

This is the activation layer, it is just a wrapper for the activation functions we defined above.

```python
class Activation():
    def __init__(self, activation, name="layer"):
        self.activation = activation.activation
        self.activation_prime = activation.gradient
        self.input = None
        self.name = name
        self.output = None
    
    def forward(self, x):
        self.input = x
        self.output = self.activation(x)
        return self.output
        
    def backward(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error
    
    def __call__(self, x):
        return self.forward(x)
```

### Linear Layer

This is the linear layer class. We define W and b because our layer is just $y=wx+b$​​. with $w$ and $b$ being matrices.

```python
class Linear():
    def __init__(self, n_in, n_out, name="layer"):
        limit = 1 / np.sqrt(n_in)
        self.W = np.random.uniform(-limit, limit, (n_in, n_out))
        self.b = np.zeros((1, n_out)) # Biases
        self.name = name
        self.input = None
        self.output = None
    
    def forward(self, x):
        self.input = x
        self.output = np.dot(self.input, self.W) + self.b
        return self.output
    
    def __call__(self, x):
        return self.forward(x)
    
    def backward(self, error, lr=0.01):
        input_error = np.dot(error, self.W.T)
        delta = np.dot(self.input.T, error)
        
        self.W -= lr * delta
        self.b -= lr * np.mean(error)
        return input_error
```

## Model

We create the model class. 

We have 3 linear layers, and 3 activation layers. The first 2 activation layers are `LeakyReLU` for the outputs of the first 2 linear layers, and layer 3 is the output layer, so we have `Softmax()` as our activation for that layer

We define a forward function where we just pass `x` through each layer and then return the result of the final layer

After that, we define a backward function, which back propagates throgh the layers calling the individual layers `backward` function

```python
class Network():
    def __init__(self, input_dim, output_dim lr = 0.01):
        self.layers = [
            Linear(input_dim, 256, name="input"),
            Activation(LeakyReLU(), name="input_relu"),
            Linear(256, 128, name="layer2"),
            Activation(LeakyReLU(), name="layer2_relu"),
            Linear(128, output_dim, name="output"),
            Activation(Softmax(), name="output_softmax")
        ]
        self.lr = lr
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
    def backward(self, loss_grad):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad, self.lr)
    
    def __call__(self, x):
        return self.forward(x)
```



## Training

### Setup

In the following segment, the model and the loss are initialized

```python
criterion = CrossEntropy()
model = NeuralNetwork(n_input_dim, n_out, lr=1e-3)
```

Now, the setup is done, lets train the model.

We create a PyTorch style training loop.

1. We split the data into batches of size 256
2. For each batch, we do the following
   1. Feed the data to the model’s `forward` method with `out = model.forward(x_batch)`
   2. Calculate the loss and accuracy for displaying
   3. Calculate the gradient of the loss through `error = criterion.gradient(y_batch, out)`
   4. Do backpropagation through the `model.backward(error)` function, where we feed in the loss function gradient.
3. After the model runs through all the batches, we print the loss and accuracy for that epoch

```python
EPOCHS = 5
with tf.device(device): ## tf.device("/device:GPU:0")
    ## BATCH THIS ##
    for epoch in range(EPOCHS):
        loss = []
        acc = []
        for x_batch, y_batch in batch_iterator(X_train, y_train, batch_size=256):
            out = model(x_batch)
            loss.append(np.mean(criterion.loss(y_batch, out)))
            acc.append(accuracy_score(np.argmax(y_batch, axis=1), np.argmax(out, axis=1)) * 100)
            error = criterion.gradient(y_batch, out)
            model.backward(error)
        if epoch % 1 == 0:
            print("Epoch: {}, Loss: {}, Acc: {}".format(epoch, np.mean(loss), np.mean(acc)))

```

```
[OUTPUT]
Epoch: 0, Loss: 0.1023689853553398, Acc: 81.88996010638297
Epoch: 1, Loss: 0.043075670774549345, Acc: 92.58643617021276
Epoch: 2, Loss: 0.03227421572072945, Acc: 94.55119680851064
Epoch: 3, Loss: 0.025909577243929914, Acc: 95.64494680851064
Epoch: 4, Loss: 0.021599316143203488, Acc: 96.43783244680851
```



## Testing

```python
out = model.forward(X_test)
accuracy_score(np.argmax(y_test, axis=1), np.argmax(out, axis=1))
```

```
0.9636
```

We get an accuracy of $96$​% on the test data, which is really good and means that our from-scratch model is learning

The following code segment shows you how to see the output for any one image.

```python
imagen = 23 # The image we want to see
print("Predicted: {}, Actual: {}".format(np.argmax(out[imagen]), np.argmax(y_test, axis=1)[imagen]))
```

```
Predicted: 5, Actual: 5
```

Here’s a bonus idea, implement a Convolutional Layer in numpy and train it on MNIST

That’s it for this tutorial, I will see you in the next one. 