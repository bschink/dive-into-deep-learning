# 📘 Chapter 05 – Multilayer Perceptrons

In this chapter, I learned how to build and train multilayer perceptrons (MLPs) for more complex learning tasks. By introducing nonlinear activation functions and hidden layers, I extended the capabilities of linear models. I implemented MLPs both from scratch and using PyTorch’s built-in layers. The chapter also covered key theoretical and practical topics such as forward and backward propagation, computational graphs, numerical stability, parameter initialization and regularization techniques like dropout and early stopping. Finally, I applied these concepts in a hands-on Kaggle project by building a model to predict house prices using k-fold cross-validation. There are no notebook for chapters 5.3 and 5.5 because there was nothing to code there.

---

## 📂 Contents

| Notebook | Topic |
|----------|-------|
| `05_01_multilayer_perceptrons` | Multilayer Perceptrons, nonlinear activation functions |
| `05_02_implementation_of_multilayer_perceptrons` | MLP implementation |
| `05_04_numerical_stability_and_initialization` | vanishing & exploding gradients, parameter initialization |
| `05_06_dropout` | dropout with implementation |
| `05_07_predicting_house_prices_on_kaggle` | predicting house prices |

---

## 📌 Topics Covered by Subchapter

### 5.1. Multilayer Perceptrons

Building Multilayer Perceptrons by introducing hidden layers and adding more capabilities than with a single layer by introducing nonlinear activation functions (ReLu, sigmoid, tanh).

### 5.2. Implementation of Multilayer Perceptrons

Implementation of MLP with one hidden layer both from scratch and using high level Pytorch functions

### 5.3. Forward Propagation, Backward Propagation, and Computational Graphs

A deeper look into forward propagation, backward propagation and computational graphs as means for visualization.

### 5.4. Numerical Stability and Initialization

Stating the problems of vanishing and exploding gradients and parameter initialization strategies to account for that.

### 5.5. Generalization in Deep Learning

Revisited overfitting and regularization and introduced early stopping as another technique for better generalization.

### 5.6. Dropout

Learned about the dropout technique and implemented it from scratch and using high level Pytorch functions.

### 5.7. Predicting House Prices on Kaggle

Creating a first submission for the predicting house prices competition on Kaggle using linear regression as a benchmark for future models. Data was downloaded, preprocessed and the model trained with k-fold cross validation.

---

➡️ Next up: [Chapter 06 – Builders’ Guide](../chapter_06_builders_guide/)
