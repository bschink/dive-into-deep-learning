# 📘 Chapter 03 – Linear Neural Networks for Regression

In this chapter, I learned how to implement and train linear neural networks for regression tasks. Starting from synthetic data generation, I explored building linear regression models from scratch as well as using PyTorch’s concise tools. I also introduced an object-oriented design for structuring models, datasets, and training loops, which will serve as a foundation for future chapters. Key machine learning concepts such as underfitting, overfitting, cross-validation, and regularization through weight decay were revisited and implemented.

---

## 📂 Contents

| Notebook | Topic |
|----------|-------|
| `03_01_linear_regression.ipynb` | models, loss functions, optimization (SGD), linear regression |
| `03_02_object-oriented_design_for_implementation` | Module, DataModule & Trainer classes |
| `03_03_synthetic_regression_data` | generate and load synthetic regression data |
| `03_04_linear_regression_implementation_from_scratch` | implement linear regression from scratch |
| `03_05_concise_implementation_of_linear_regression` | implementation of linear regression using all Pytorch tools |
| `03_06_generalization` | underfitting, overfitting, cross validation |
| `03_07_weight_decay` | regularization in form of weight decay |

---

## 📌 Topics Covered by Subchapter

### 3.1. Linear Regression

Revised the concepts of model, loss function, optimization via SGD and linear regression.

### 3.2. Object-Oriented Design for Implementation

Introduced an object-oriented structure for the further implementation of models using three base classes (Module, DataModule, Trainer).

### 3.3. Synthetic Regression Data

To confirm that the implementation of a particular learning algorithm works as expected it can be helpful to test it with synthetic data. Therefore this chapter shows how to implement a synthetic data generator for linear regression and how to read and load the generated data.

### 3.4 Linear Regression Implementation from Scratch

Implementing linear regression (model, loss function, optimization algorithm) from scratch only using Pytorch for tensors and automatic differentiation for computing gradients.

### 3.5 Concise Implementation of Linear Regression

Implementing linear regression using all the features Pytorch offers like Lazy Linear, MSELoss and SGD.

### 3.6 Generalization

Revision on training and generalization error, underfitting and overfitting and k-fold cross validation.

### 3.7 Weight Decay

Revised regularization and learned about the differences using it in neural networks opposed to classical machine learning.

---

➡️ Next up: [Chapter 04 – Linear Neural Networks for Classification](../chapter_04_linear_neural_networks_for_classification/)
