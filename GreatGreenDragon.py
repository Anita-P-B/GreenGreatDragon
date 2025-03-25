# install necessary libraries
!pip install transformers torch --quiet

# import necessary libraries
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from itertools import permutations
import torch.nn.functional as F
import numpy as np

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")
model.eval()

# get all possible permutations out of input text
def permutation_function(input_sentence):
  '''
  receives a sentence.
  return all possible permutations of words inside of the sentence  
  '''
  string_arr = input_sentence.split(" ")
  perms = list(permutations(string_arr))
  return perms

# evaluate given permutation by giving probability score
def eval_permutation(input_sentence, tokenizer, model):
    '''
    Receives a sentence.
    Returns a score of how much the sentence "makes sense" to the language model,
    based on the product of the predicted probabilities for each next token.
    '''
    inputs = tokenizer(input_sentence, return_tensors="pt")
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # shape: [1, seq_len, vocab_size]

    total_log_prob = 0.0

    # Loop over each token position (except the last, since there's nothing after it)
    for i in range(1, logits.size(1)):
        # Get predicted distribution for token i
        probs = F.softmax(logits[0, i-1 ], dim=-1)
        
        # Get actual next token ID
        target_token_id = input_ids[0, i].item()
       
        # Get the probability assigned to the actual next token
        token_prob = probs[target_token_id].item()
     
        # Accumulate log-probability
        total_log_prob += torch.log(torch.tensor(token_prob + 1e-8))  # add epsilon for stability

    return total_log_prob.item()  # higher = more likely

# evaluate all possible permutations and print score table
def best_permutation(input_text):
    perms = permutation_function(input_text)
    score_arr = np.empty(len(perms))
    
    for i in range(len(perms)):
        sentence = ' '.join(perms[i])
        score_arr[i] = eval_permutation(sentence, tokenizer, model)

    # Zip and sort results by score (descending)
    results = list(zip(perms, score_arr))
    results.sort(key=lambda x: x[1], reverse=True)

    print(f"The best permutation for the sentence '{input_text}' is:")
    print(f"  -> '{' '.join(results[0][0])}' (Score: {results[0][1]:.4f})\n")
    
    print("All permutations ranked:")
    for i, (perm, score) in enumerate(results, 1):
        sentence = ' '.join(perm)
        print(f"{i:>2}. {sentence:<30} -> Score: {score:.4f}")

    return results
