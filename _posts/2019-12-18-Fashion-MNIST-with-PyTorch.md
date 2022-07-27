---
title: "MNIST with Tensorflow"
date: 2019-12-17
toc: true
toc_label: "Contents"
excerpt: "Classify the MNIST dataset with Tensorflow and Python"
categories:
  - Programming
  - Machine Learning
tags:
  - Python
  - Computer Vision
  - Tensorflow
---
Tensorflow is a powerful deep learning framework developed by the team at Google. It is used to build highly accurate neural networks.

This is probably my last post of the year (for future readers, the year that I am writing this in is 2019).

We will use Tensorflow (it's Keras API) to train a model that can classify the MNIST Dataset.

> **Stay Tuned** because I plan on making a tutorial for classifying images with Tensorflow's subclassing API and PyTorch (which is another deep learning framework like tensorflow)

Let's get started

## Getting started

You need to install tensorflow. If you already have tensorflow installed, you can skip this.

To install tensorflow, run 

```sh
pip install tensorflow
```

or

```sh
conda install tensorflow # For anaconda
```

## Getting the data

In your program or notebook, add the following code:

```python
import tensorflow as tf
mnist = tf.keras.datasets.mnist # Getting the MNIST

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0 
```

## Building the model

Add the following code to your program

```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])
```

This code creates the model. The first layer flattens the image for the model. The second layer is a hidden layer in which the learning takes place. The third layer is a dropout layer to prevent overfitting. The final layer returns the class that the model believes it to be. There are 10 neurons for 10 classes.

### Compiling the model

When making tensorflow models, you need to compile it. Here is the code:

```python
model.compile(optimizer='adam' loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

We can now train the model

## Training the model

Training the model in tensorflow is very simple. It requires only 1 line of code. Here is the code:

```python
model.fit(x_train, y_train, epochs=5)
```

This may take a bit of time so you are going to have to wait.

## Evaulating the model

Now that the model is done training, we can evaulate the model. Evaulating the model is also very simple in tensorflow just like everything else.

```python
model.evaluate(x_test, y_test) # Only 1 line
```

That's it. We have built a model in tensorflow that can classify the MNIST dataset.

## Conclusion

This was very simple and as you can see, you can get started with tensorflow in around 5 minutes. It is that easy to work with tensorflow. Tensorflow has another API called the SubClassing API which is more complicated but allows you to build models that are more powerful. There are also other Deep Learning frameworks like PyTorch which helps you to build models but this is the easiest to learn  and implement. Have fun with your new model and I will see you next year!