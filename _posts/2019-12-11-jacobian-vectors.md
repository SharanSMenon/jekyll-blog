---
title: "Calculating Jacobian Vectors"
date: 2019-12-11
toc: true
toc_label: "Table of contents"
categories:
  - Mathematics
tags:
  - Linear Algebra
  - Calculus
---

Jacobian vectors are a really powerful tool in mathematics, and it is very useful in machine learning. Learning how to calculate Jacobian vectors will be important to understand machine learning, so let's learn how to calculate it.

You need to know what a vector is and how to take the partial derivative of a function

## Calculating jacobian vectors

Lets calculate the jacobian vector for the following function:

$$
f(x,y,z)=3x^2+y^2z+cos(z)x
$$

This is a pretty simple function. Lets take the partial derivatives of the function first:

$$
\frac{\partial f}{\partial x} = 6x+cos(z)
$$

$$
\frac{\partial f}{\partial y} = 2yz
$$

$$
\frac{\partial f}{\partial z} = y^2-sin(z)x
$$

The Jacobian vector for a function is

$$
J= \begin{bmatrix} 
\frac{\partial f}{\partial x} & \frac{\partial f}{\partial y} & \frac{\partial f}{\partial z}
\end{bmatrix}
$$

Let's substitute our partial derivatives in to the Jacobian Vector

$$
J= \begin{bmatrix} 
6x+cos(z) & 2yz & y^2-sin(z)x
\end{bmatrix}
$$

And there you go, we have a Jacobian vector for our function.

## Evaulating the vector

We can evaluate the vector too at a certain $x$, $y$, $z$

Let's evaulate $J(0,0,0)$

We just substitute in the $x$, $y$, and $z$ values into the partial derivatives in the vector and evaluate.

$$
J=\begin{bmatrix}
6*0+cos(0)&2*0*0&sin(0)*0
\end{bmatrix}
$$

This equals

$$
J=\begin{bmatrix}
1&0&0
\end{bmatrix}
$$

## Conclusion

So you now have calculated the Jacobian vector for a function $f(x,y,z)$ and evaluated it at those three points. There are many applications to Jacobian vectors like Jacobian Matrices. It is used in machine learning and many other fields. 

So try to calculate and evaluate a Jacobian vector for a function now that you know how to calculate Jacobian vectors.

## More resources

If you are looking to learn more, here is a curated list of resources that I recommend you use.

1. [Mathematics for Machine Learning - Coursera](https://www.coursera.org/specializations/mathematics-machine-learning)
2. [Khan Academy Calculus](https://www.khanacademy.org/math/ap-calculus-bc)
3. [Wikipedia](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant)

Well, folks, that's it for this tutorial and I will see you next time.