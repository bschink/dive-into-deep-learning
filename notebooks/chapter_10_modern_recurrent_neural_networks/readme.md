# 📘 Chapter 10 – Modern Recurrent Neural Networks

In this chapter, I explored modern advancements in recurrent neural networks designed to address the limitations of vanilla RNNs. I learned about LSTMs and GRUs, which use gating mechanisms to better capture long-range dependencies in sequences. The chapter introduced techniques like stacking multiple RNN layers (deep RNNs) and processing input in both directions (bidirectional RNNs) to enhance model capacity. I also studied the encoder-decoder architecture, which forms the basis of many sequence-to-sequence tasks like machine translation. Using a French-English dataset, I implemented a full translation model with GRUs. Finally, I learned about decoding strategies such as greedy and beam search for generating translations more effectively. No notebook for 10.8. Beam Search because there was nothing to code.

---

## 📂 Contents

| Notebook | Topic |
|----------|-------|
| `10_01_long_short-term_memory_lstm` | LSTM, input, forget & output gate |
| `10_02_gated_recurrent_units_gru` | GRU |
| `10_03_deep_recurrent_neural_networks` | multilayer RNNs |
| `10_04_bidirectional_recurrent_neural_networks` | bidirectional RNNs |
| `10_05_machine_translation_and_the_dataset` | machine translation dataset |
| `10_06_the_encoder_decoder_architecture` | encoder-decoder architecture |
| `10_07_sequence_to_sequence_learning_for_machine_translation` | sequence-to-sequence learning |

---

## 📌 Topics Covered by Subchapter

### 10.1. Long Short-Term Memory (LSTM)

Introducing LSTMs and their input, forget and output gates.

### 10.2. Gated Recurrent Units (GRU)

Introducing GRUs and their reset and update gates.

### 10.3. Deep Recurrent Neural Networks

Introducing deep recurrent neural networks by stacking multiple single layer RNNs on top of each other.

### 10.4. Bidirectional Recurrent Neural Networks

Introducing bidirectional recurrent neural networks.

### 10.5. Machine Translation and the Dataset

Downloading and preprocessing a machine translation dataset which consists of french and english.

### 10.6. The Encoder–Decoder Architecture

Introducing the interfaces of the encoder-decoder architecture.

### 10.7. Sequence-to-Sequence Learning for Machine Translation

Building a sequence-to-sequence model with GRUs for machine translation.

### 10.8. Beam Search

Comparing greedy search, exhaustive search and beam search.

---

➡️ Next up: [Chapter 11 – Attention Mechanisms and Transformers](../chapter_11_attention_mechanisms_and_transformers/)
