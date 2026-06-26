"""
PyTorch Tensor Fundamentals — Exercise Set
============================================

HOW TO USE THIS FILE
---------------------
This is a plain Python script, not a notebook — but it's organized the same way
a notebook would be: numbered exercises, each with some PROVIDED code that you
run, and some parts that ask YOU to either (a) explain output in a comment, or
(b) write your own code.

For each exercise:
  - "Provided code" sections will run as-is. Run the file (or copy a section
    into a Python shell) and look at the printed output.
  - "Qn:" comments ask you to explain something about that output. Write your
    answer directly below, replacing "(write your answer here)".
  - "TODO" comments ask you to write code. Write it directly below the TODO,
    in place of "# your code here". The file should still run top-to-bottom
    after you fill these in.

We set a random seed so everyone's "random" tensors are identical — this
makes it possible to discuss specific numbers in class.

GOAL
----
By the end of this set you'll be comfortable with the shape reasoning needed
to read PyTorch code for training language models, where tensors commonly
look like (batch_size, seq_len, hidden_dim) or, inside attention,
(batch_size, num_heads, seq_len, head_dim).
"""

import torch
torch.manual_seed(42)

# =====================================================================
# Exercise 1: 4D tensors and the shapes used in attention
# =====================================================================
# In a transformer attention layer, Q and K tensors often have shape:
#     (batch_size, num_heads, seq_len, head_dim)

# --- Provided code ---
batch_size, num_heads, seq_len, head_dim = 2, 3, 5, 4
Q = torch.randint(-5, 6, (batch_size, num_heads, seq_len, head_dim)).float()
K = torch.randint(-5, 6, (batch_size, num_heads, seq_len, head_dim)).float()

K_t = K.transpose(-2, -1)
scores = Q @ K_t
print("Q.shape:", Q.shape)
print("K.shape:", K.shape)
print("K_t.shape (after transpose(-2, -1)):", K_t.shape)
print("Scores shape: ", scores.shape)

# Q1: K.transpose(-2, -1) swaps the last two dimensions, turning shape
# (2, 3, 5, 4) into (2, 3, 4, 5). The first two dimensions are untouched.
# Why do we want to leave the batch and head dimensions alone here, and only
# swap the seq_len/head_dim pair?  
#
# A1: (write your answer here)

# Q2: In the scores tensor, explain what each
# of the four dimensions represents, and specifically what a single value
# of scores, say scores[0, 1, 2, 4] means in plain English, including the meaning
# of each index.  What is the attention relationship between the 2nd and 3rd 
# words in the first attention head, in the second batch?
#
# A2: (write your answer here)
#
# Q3: How far apart in memory are K_t[1,2,3,4] and K_t[1,2,4,4]? Think carefully
# about your answer, and explain your reasoning. 
# What about K_t[1,2,3,3] and K_t[1,2,3,4]?  How far apart are they?
#
# A3: 
