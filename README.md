# 🐉 Great Green Dragon

A tiny, fun NLP project that explores **words order preferences** using a pre-trained language model. This project does not explicitly recognize grammatical roles (like adjectives or nouns). Instead, it treats all input words equally and searches for the most likely ordering based on a language model’s internal knowledge of how natural phrases are typically constructed.

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

bash
'''pip install transformers torch numpy```

## 🚀 Run the Script

from great_green_dragon import best_permutation

best_permutation("The green great dragon")

## ✨ Example Output

The best permutation for the sentence 'The green great dragon' is:
  -> 'dragon The great green' (Score: -17.5732)

All permutations ranked:
 1. dragon The great green         -> Score: -17.5732
 2. green The great dragon         -> Score: -18.3574
 3. great The green dragon         -> Score: -18.3897
 4. dragon The green great      -> Score: -22.1798
 5. great The dragon green         -> Score: -23.1762



## 📚 Background

This project was inspired by the idea that certain adjective orders “sound right” due to linguistic patterns learned over time. The code demonstrates how a language model—trained only to predict the next word—has implicitly learned those patterns.

---
May your words find their order — and your prompts find their power.



