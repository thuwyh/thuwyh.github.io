---
title: "Logistic Regression: A Beginner's Guide"
categories: 
    - ML101
tags: 
    - "Logistic Regression"
    - "Machine Learning"
---

Logistic Regression is a statistical method used to analyze the relationship between a categorical dependent variable and one or more independent variables. It is widely used in machine learning and predictive modeling for binary classification problems. In this article, we will discuss the basics of logistic regression and its mathematical formulation.

## Binary Classification

Binary classification is a type of classification problem in which the output variable can take only two possible values, usually represented as 0 or 1. For example, predicting whether a given email is spam or not spam, or predicting whether a person is likely to purchase a product or not.

## The Sigmoid Function

The logistic regression model is based on the sigmoid function which maps any real-valued number to a value between 0 and 1. The sigmoid function is defined as follows:

$$\sigma(z) = \frac{1}{1+e^{-z}}$$

where z is the input to the sigmoid function. The sigmoid function has an S-shaped curve and is used to convert any input value to a probability value between 0 and 1. In logistic regression, we use the sigmoid function to predict the probability of the output variable being 1, given the input variables.

## The Logistic Regression Model

The logistic regression model can be expressed as follows:

$$P(y=1|x) = \sigma(\beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n)$$

where x is the input variables, y is the output variable, and $\beta$ is the coefficient vector. The logistic regression model predicts the probability of the output variable being 1, given the input variables.

## Maximum Likelihood Estimation

The coefficients of the logistic regression model are estimated using Maximum Likelihood Estimation (MLE). MLE is a statistical method used to estimate the parameters of a model by maximizing the likelihood of the observed data. In the case of logistic regression, the likelihood function is given by:

$$L(\beta) = \prod_{i=1}^{n} P(y_i|x_i;\beta)^{y_i}(1 - P(y_i|x_i;\beta))^{1-y_i}$$

where $y_i$ is the observed output for the ith input $x_i$. The goal of MLE is to find the values of $\beta$ that maximize the likelihood function. This is typically done using numerical optimization techniques such as gradient descent.

## Conclusion

Logistic Regression is a powerful statistical method used for binary classification problems. It is widely used in machine learning and predictive modeling due to its simplicity and interpretability. In this article, we discussed the basics of logistic regression and its mathematical formulation. We hope this article provides a good starting point for those interested in learning more about logistic regression.