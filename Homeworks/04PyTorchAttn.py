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
import torch.nn.functional as F

torch.manual_seed(42)


# =====================================================================
# Exercise 1: Creating tensors and reading their shape
# =====================================================================

# --- Provided code ---
T2D = torch.randint(-5, 6, (3, 4))   # 3 rows, 4 columns
T3D = torch.randint(-5, 6, (2, 3, 4))  # 2 "blocks" of (3, 4)

print("T2D:\n", T2D)
print("T2D.shape:", T2D.shape, " ndim:", T2D.ndim)
print()
print("T3D:\n", T3D)
print("T3D.shape:", T3D.shape, " ndim:", T3D.ndim)

# Q1: T2D.shape is (3, 4) and T3D.shape is (2, 3, 4). Using just t#d and
# appropriate indexing, produce tensors that have shape (3,4) like T2D, and
# shape (4). 
#
# A1: (write your answer here)


# =====================================================================
# Exercise 2: Reductions along a dimension
# =====================================================================

# --- Provided code ---
print("T2D.sum():", T2D.sum())
print("T3D.sum(dim=1):\n", T3D.sum(dim=1))
print("T3D.sum(dim=1).shape:", T3D.sum(dim=1).shape)
print("T3D.mean(dim=1):\n", T3D.float().mean(dim=1))

# Q1: T3D.sum(dim=1) has shape (2, 4) — one dimension of T3D "disappeared".
# Which dimension was it, and why does that leave you with shape (2, 4)
# instead of, say, (2, 3) or (3, 4)?
#
# A1: (write your answer here)

# Q2: We had to call T3D.float() before .mean(dim=1) — T3D.mean(dim=1) on
# its own raises a RuntimeError. Check T3D.dtype to see why, and explain in
# your own words why PyTorch refuses to average integer tensors directly.
#
# A2: (write your answer here)

# TODO 1: Write code that computes the sum of T3D along each row instead of each column.
# Before running it, write your PREDICTED shape as a comment. Then print the
# actual shape and confirm.
#
# Predicted shape: (write your prediction here)
# your code here


# =====================================================================
# Exercise 3: Broadcasting — vector against a matrix
# =====================================================================

# --- Provided code ---
vec4 = torch.randint(-5, 6, (4,))
result = T2D + vec4
print("vec4:", vec4)
print("T2D + vec4:\n", result)
print("result.shape:", result.shape)

# Q1: T2D has shape (3, 4) and vec4 has shape (4,) — these are different
# shapes, yet the addition worked. Explain, dimension by dimension, how
# PyTorch lined these shapes up to figure out what "+" should mean here.
#
# A1: (write your answer here)

# TODO 1: Create a tensor of shape (3, 1) with random small integers, add it
# to T2D, and print the result and its shape.
# your code here

# Q2: Compare the result of adding your (3, 1) tensor to T2D versus adding
# vec4 (shape (4,)) to T2D. The two results spread their added values across
# T2D very differently. Explain what's different and why, in terms of which
# axis got "stretched".
#
# A2: (write your answer here)


# =====================================================================
# Exercise 4: Broadcasting — matrix against a 3D tensor
# =====================================================================

# --- Provided code ---
mat34 = torch.randint(-5, 6, (3, 4))
result3d = T3D + mat34
print("T3D + mat34:\n", result3d)
print("result3d.shape:", result3d.shape)

# Q1: mat34 has shape (3, 4), the same as a single "slice" of T3D. Explain
# what happened to mat34 conceptually for this addition to produce a (2, 3, 4)
# result — what got applied to each of the 2 slices of T3D?
#
# A1: (write your answer here)

# TODO 1: Try adding a tensor of shape (2, 4) to T3D (shape (2, 3, 4)).
# This should raise a RuntimeError. Write the code, run it, then paste the
# key line of the error message as a comment below your code.
# your code here
#
# Error message (paste relevant line here): (write it here)
#
# Q2: Using PyTorch's broadcasting rule (compare shapes from the rightmost
# dimension outward; dimensions are compatible if equal, or if one of them
# is 1), explain specifically why (2, 4) fails to broadcast against
# (2, 3, 4), while (3, 4) and (4,) both succeeded.
#
# A2: (write your answer here)

# =====================================================================
# Exercise 5: Element-wise multiplication vs. matrix multiplication
# =====================================================================

A = torch.randint(-5, 6, (3, 4))
B = torch.randint(-5, 6, (3, 4))
C = torch.randint(-5, 6, (4, 3))

elementwise = A * B
matmul_result = A @ C  # equivalent to torch.matmul(A, C)

print(f"A:{A}\nB:{B}\nC:{C}\n")
print("A * B (element-wise):\n", elementwise, "\nshape:", elementwise.shape)
print("A @ C (matrix multiply):\n", matmul_result, "\nshape:", matmul_result.shape)

# Q1: A * B requires A and B to have the SAME shape (3, 4), and the result is
# also (3, 4). A @ C requires A's shape (3, 4) and C's shape (4, 3) to share
# a specific dimension, and the result is (3, 3). Explain the rule for which
# dimensions must match in each case, and why the two operations produce such
# differently-shaped results.  For each operation, how many individual multiplications
# are performed?  How many additions?
#
# A1: (write your answer here)

# TODO 1: Write code that matrix-multiplies C by a new # random tensor T 
# of a shape such that the matrix-multiply requires a total of 24
# multiplies and 16 adds.  Print the result and its shape.
# your code here

# =====================================================================
# Exercise 6: Batched matrix multiplication
# =====================================================================

# TODO 1: Create random W with shape (4, 5) — and compute T3D @ W2. 
# Print the result's shape.
# your code here

# Q1: W has shape (4, 5), not (2, 4, 5), yet T3D @ W still works 
# Explain what combination of broadcasting AND matrix multiplication rules 
# makes this possible, and specifically, how many multiplies and adds are
# performed in the operation.
#
# A1: (write your answer here)


# =====================================================================
# Exercise 7: Reshaping a tensor
# =====================================================================

# --- Provided code ---
flattened = T3D.reshape(2, 12)
regrouped = T3D.reshape(6, 4)

print("T3D.shape:", T3D.shape)
print("flattened.shape:", flattened.shape)
print("regrouped.shape:", regrouped.shape)

# Q1: Explain why reshape(2, 12) and reshape(6, 4) are both valid, but 
# reshape(2, 10) would raise an error.
#
# A1: (write your answer here)

# Q2: regrouped has shape (6, 4). Conceptually, which two of T3D's original
# dimensions got "merged" together to produce that leading 6, and in what
# order are the original 3-row blocks laid out within those 6 rows?
# (Hint: print regrouped and compare its rows to T3D's two (3,4) slices.)
#
# A2: (write your answer here)

# =====================================================================
# Exercise 8: 4D tensors and the shapes used in attention
# =====================================================================
# In a transformer attention layer, which we'll soon examine, tensors often have
# shape:
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

# Q1: In the scores tensor, explain what each
# of the four dimensions represents, and specifically what a single value
# of scores, say scores[0, 1, 2, 4] means in plain English, including the meaning
# of each index.  What is the attention relationship between the 2nd and 3rd 
# words in the first attention head, in the second batch?
#
# A1: (write your answer here)


# =====================================================================
# Exercise 9 (capstone): Scaled dot-product attention
# =====================================================================
# This puts everything together — reductions, broadcasting, batched matmul,
# transpose, and softmax — into the core computation used in every
# transformer layer.

# --- Provided code (scaffold; you complete the TODOs inside) ---
V = torch.randint(-5, 6, (batch_size, num_heads, seq_len, head_dim)).float()

def scaled_dot_product_attention(Q, K, V):
    """
    Q, K, V: tensors of shape (batch_size, num_heads, seq_len, head_dim)
    Returns: output tensor of shape (batch_size, num_heads, seq_len, head_dim)
    """
    head_dim = Q.shape[-1]

    # TODO 1: Compute raw attention scores: Q @ K^T over the last two dims.
    # scores should have shape (batch_size, num_heads, seq_len, seq_len)
    scores = None  # your code here

    # TODO 2: Scale the scores by dividing by sqrt(head_dim).
    scaled_scores = None  # your code here

    # TODO 3: Apply softmax along the correct dimension so each row of
    # scaled_scores sums to 1.
    attn_weights = None  # your code here

    # TODO 4: Multiply attn_weights by V to get the final output.
    output = None  # your code here

    return output

# Uncomment once the TODOs above are filled in:
# output = scaled_dot_product_attention(Q, K, V)
# print("output.shape:", output.shape)

# Q1: Why do we divide by sqrt(head_dim) before the softmax, rather than
# feeding the raw Q @ K^T scores directly into softmax?
#
# A1: (write your answer here)

# Q2: We applied softmax along dim=-1. What would it mean, conceptually, if
# we'd mistakenly applied it along dim=-2 instead? Would the shape change?
# Would the meaning of the result change?
#
# A2: (write your answer here)

# Q3: Stepping back — for LLM training, why is it useful that this entire
# function works on a BATCH of sequences (the leading batch_size dimension)
# at once, rather than being written to handle one sequence at a time and
# called in a loop?
#
# A3: (write your answer here)
