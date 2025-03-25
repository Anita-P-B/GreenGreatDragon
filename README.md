# 🐉 Great Green Dragon

A tiny, fun NLP project that explores **adjective order preferences** using a pre-trained language model.

Inspired by a linguistic observation from Tolkien—why does *"great green dragon"* sound more natural than *"green great dragon"*?

## 🔍 What It Does

Given a phrase with two adjectives and a noun (e.g., `"green great dragon"`), this script:

1. Generates **all possible word order permutations**
2. Uses a **language model (DistilGPT2)** to score how natural each permutation sounds
3. Outputs the **most natural-sounding phrase**, according to the model
4. Displays a ranked list of all permutations and their scores

## 🧠 How It Works

The core logic:
- Tokenizes each phrase and feeds it to a causal language model
- Uses log-probabilities to evaluate how well the model "expects" the next word at each position
- Ranks permutations by their total sentence likelihood

## 🛠️ Requirements

- Python 3.7+
- `transformers`
- `torch`
- `numpy`

You can install them via pip:

```bash
pip install transformers torch numpy
