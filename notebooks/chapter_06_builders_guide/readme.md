# 📘 Chapter 06 – Builders’ Guide

In this chapter, I explored PyTorch’s core building blocks in more depth, focusing on writing clean, modular, and flexible deep learning code. I learned how to create custom modules and layers, manage and initialize parameters, and implement lazy initialization. The chapter also covered practical aspects like saving and loading model parameters, as well as leveraging GPUs for computation.

---

## 📂 Contents

| Notebook | Topic |
|----------|-------|
| `06_01_layers_and_modules` | custom Pytorch modules, Sequential module from scratch |
| `06_02_parameter_management` | parameter access, tied parameters |
| `06_03_parameter_initialization` | built-in & custom initializers |
| `06_04_lazy_initialization` | lazy initialization |
| `06_05_custom_layers` | custom layers with/without parameters |
| `06_06_file_io` | loading & saving tensors |
| `06_07_gpus` | computing devices |

---

## 📌 Topics Covered by Subchapter

### 6.1. Layers and Modules

Introduction to building custom Pytorch modules and different ways of using them. Additionally the function of the Sequential module was explained by implementing it from scratch.

### 6.2. Parameter Management

How to access parameters inside modules and how to tie paramters of different layers together.

### 6.3. Parameter Initialization

How to initialize parameters using built-in and custom initializers.

### 6.4. Lazy Initialization

Explained how lazy initialization inside Pytorch works

### 6.5. Custom Layers

Building custom layers with or without parameters.

### 6.6 File I/O

How to load and save tensors and model parameters to files.

### 6.7. GPUs

Explained how to create computing devices and how to copy or store tensors on the gpu.

---

➡️ Next up: [Chapter 07 – Convolutional Neural Networks](../chapter_07_convolutional_neural_networks/)
