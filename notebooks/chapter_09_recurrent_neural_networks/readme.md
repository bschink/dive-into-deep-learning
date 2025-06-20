# 📘 Chapter 09 – Recurrent Neural Networks

In this chapter, I learned how to model and process sequential data using recurrent neural networks (RNNs). The chapter began with an introduction to sequence modeling and language modeling concepts such as Markov models, n-grams, Laplace smoothing, and perplexity. I learned how to tokenize raw text and prepare it as input for RNNs. Then I implemented a basic RNN from scratch, handling challenges like exploding gradients through gradient clipping, and later re-implemented it using PyTorch’s built-in layers. The chapter concluded with a theoretical explanation of backpropagation through time (BPTT). No notebook for 9.7 because there was nothing to code.

---

## 📂 Contents

| Notebook | Topic |
|----------|-------|
| `09_01_working_with_sequences` | autoregressive models, sequence models |
| `09_02_converting_raw_text_into_sequence_data` | tokenization, vocabulary |
| `09_03_language_models` | markov models, laplace smoothing, perplexity |
| `09_04_recurrent_neural_networks` | recurrent neural networks |
| `09_05_recurrent_neural_network_implementation_from_scratch` | rnn from scratch, gradient clipping |
| `09_06_concise_implementation_of_recurrent_neural_networks` | rnn with pytorch |

---

## 📌 Topics Covered by Subchapter

### 9.1. Working with Sequences

Explaining autoregressive and sequence models and introducing markov models.

### 9.2. Converting Raw Text into Sequence Data

How to read the dataset, tokenize it, defining a vocabulary and putting it all together.

### 9.3. Language Models

Introducing language models e.g. markov models and n-grams and also laplace smoothing and the perplexity measure.

### 9.4. Recurrent Neural Networks

Introducing recurrent neural networks.

### 9.5. Recurrent Neural Network Implementation from Scratch

Implementing recurrent neural networks without high-level API functions and introducing gradient clipping.

### 9.6. Concise Implementation of Recurrent Neural Networks

Implementing recurrent neural networks with high-level Pytorch functions.

### 9.7 Backpropagation Through Time

Explaining how to do backpropagation on recurrent neural networks

---

➡️ Next up: [Chapter 10 – Modern Recurrent Neural Networks](../chapter_10_modern_recurrent_neural_networks/)
