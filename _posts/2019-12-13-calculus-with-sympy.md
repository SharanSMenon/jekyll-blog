---
title: "Calculus with **Sympy**"
date: 2019-12-13
toc: true
toc_label: "Topics"
excerpt: "Do calculus with Sympy in Python"
categories:
  - Mathematics
  - Programming
tags:
  - Python
  - Calculus
---

There is a python library called [SymPy](https://www.sympy.org/en/index.html). It is a python library for symbolic mathematics (means that it can work with symbols like x, y, and z). This library can do calculus which means that it can find limits, derivatives, and integrate functions. It can also do many other things like solve matrices, vectors, etc. This tutorial will show you how to use SymPy

> **Note**: Before continuing on with this tutorial, make sure that you are familiar with calculus and its concepts. The topics covered in this article will be limits, derivatives, and integrals. You can head over to Khan Academy or get a calculus course if you don't know what that is or if you just need a refresher.

## Installing Sympy

In this short section, we will just install sympy. If you don't already have sympy installed, just run the following command in your terminal:

```
pip install sympy
```

> **Note**: If you do not want to deal with installing and importing sympy (which is very easy), you can try out the sympy live shell. Just head over to sympy.org and you will see it there

## Getting Started with Sympy

Once you have sympy installed, import it into your notebook

```py
from sympy import * # Just importing everything
```

Now that we have sympy imported, we can do some calculus with it.

We need to create some variables to do calculus with so add this to your code:

```python
x, y, z = symbols('x y z') # We are making mathematical symbols
```

> We will probably not need $y$ and $z$ but we have it there just in case or if you want to use it.

## Limits

We can finally get to the calculus portion. We are going to first find the limits of functions and then we will move on to derivatives and integrals. Let's get started!

Let's calculate the limit of the function $3x^2 +2x$ at $x=0$. We know the answer is $0$.

Here is the python code:

```python
limit(3*x**2+2*x, x, 0) # Should give 0
```

As you can see, using Sympy to find limits is very simple and easy.

Let's move on to finding derivatives.

## Derivatives

In this section, we will learn to find the derivative of functions and how to use them. 

We will find the derivative of the function $x^2$. We can calculate the derivative by hand which we know to be $2x$ thanks to the power rule.

Let's do this in sympy. Here is the code for finding the derivative:

```python
derivative = diff(x**2, x)
```

If you want a function that you can evaulate, say you want to find the instantaneous rate of change at $x$, then you can lambdify it. Here's how you do it:

```python
f = lambdify(x, derivative)
print(f(2)) # Prints 4 since 2*2 = 4
```

Let's move on to integrating functions.

## Integrals

In this last section, we will learn how to use integrals with sympy.

> Integrals are also known as antiderivatives.

Let's integrate the following function:

$$
f(x)=3x^2+3x
$$

We know that the answer is this:

$$
\int f(x)dx=x^3+
\frac{3}{2}x^2
$$

Let's do this with sympy:

```
integrate(3*x**2 + 3*x)
```

If you want to lambdify the function, just follow the same steps shown in the derivative section and you can evaulate it.

## Conclusion

You have learned how to differentiate, integrate, and find the limits of functions. These are very important topics in calculus and you will use this a lot throught your life. Calculus is at the heart of machine learning and deep learning. It is also very important for physics, and countless other subjects so learning it is important.