"""
Word Embeddings & Tokenization — Exercise Set
================================================

HOW TO USE THIS FILE
---------------------
Same format as the tensor fundamentals exercise: run the "Provided code"
sections, fill in the "TODO" code sections, and answer the "Qn:" questions
directly below each one.

A NOTE ON HOW THESE QUESTIONS ARE GRADED
------------------------------------------
Every question in this file has a single, concrete, checkable answer — a
number, a shape, a True/False, a specific word, a specific index. None of
them ask you to "explain" anything in prose. If you can't get the exact
answer, that's a sign to re-read the code above the question, not to write
a paragraph around the uncertainty. Most questions follow a "predict, then
verify" pattern: work out the answer from the code/output above before you
run the next line, then confirm you were right.

We fix random seeds throughout so your numbers should match a reference
run almost exactly (sampling-based questions may vary by a percentage
point or two depending on your PyTorch version — note this where relevant).
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter

torch.manual_seed(42)

# =====================================================================
# EXERCISE 1: Token Stream -> DataLoader -> Embedding Layer
# =====================================================================
# This is the data pipeline shape every LLM training script starts with:
# raw text -> integer token ids -> batched windows -> embedding vectors.

# --- Provided code: a. tiny corpus and a word-level tokenizer ---
corpus = (
    "99 bottles of beer on the wall, 99 bottles of beer. "
    "Take one down, pass it around, 98 bottles of beer on the wall. "
    "98 bottles of beer on the wall, 98 bottles of beer. "
    "Take one down, pass it around, 97 bottles of beer on the wall. "
    "97 bottles of beer on the wall, 97 bottles of beer. "
    "Take one down, pass it around, 96 bottles of beer on the wall."
)

words = corpus.split()
vocab = sorted(set(words))
word2id = {w: i for i, w in enumerate(vocab)}
id2word = {i: w for w, i in word2id.items()}
token_ids = [word2id[w] for w in words]

print("number of words in corpus:", len(words))
print("vocab size:", len(vocab))

# Q1: Before printing it, predict len(token_ids) given len(words) above.
# Then print token_ids and confirm. What is the exact value?
#
# A1 (predicted, then confirmed): (write the exact integer here)

# --- Provided code: b. a windowed Dataset over the token stream ---
class TokenWindowDataset(Dataset):
    """Each example is a (x, y) pair of length block_size, where y is x
    shifted one position later — the standard next-token training setup."""
    def __init__(self, ids, block_size):
        self.ids = ids
        self.block_size = block_size

    def __len__(self):
        return len(self.ids) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.ids[idx: idx + self.block_size])
        y = torch.tensor(self.ids[idx + 1: idx + self.block_size + 1])
        return x, y

block_size = 6
dataset = TokenWindowDataset(token_ids, block_size)

# Q2: Using the formula in __len__ (len(ids) - block_size) and the value
# you found in Q1, predict len(dataset) WITHOUT running it. Write your
# prediction, then print len(dataset) and confirm.
#
# A2 (predicted, then confirmed): (write the exact integer here)

batch_size = 8
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# Q3: With drop_last=True, a DataLoader yields exactly len(dataset) //
# batch_size full batches and discards any remainder. Using your answer to
# Q2, predict how many batches this dataloader will produce in one epoch.
# Then confirm with len(dataloader).
#
# A3 (predicted, then confirmed): (write the exact integer here)

# --- Provided code: c. pull one batch and embed it ---
xb, yb = next(iter(dataloader))
print("xb.shape:", xb.shape)
print("yb.shape:", yb.shape)

embed_dim = 16
embedding = torch.nn.Embedding(len(vocab), embed_dim)
out = embedding(xb)
print("out.shape:", out.shape)

# Q4: out.shape has three dimensions. State exactly what each of the three
# numbers equals, using the variable names above (e.g. "dim 0 = batch_size
# because ..." is NOT what we want — we want: "dim 0 = 8", "dim 1 = 6",
# "dim 2 = 16"). Just the three numbers, in order.
#
# A4: (write the three numbers here, in order)

# Q5: How many trainable parameters does this `embedding` layer have?
# Compute it as an exact integer using len(vocab) and embed_dim (don't just
# guess — verify with sum(p.numel() for p in embedding.parameters())).
#
# A5: (write the exact integer here)

# TODO 1: Without changing embed_dim, find the exact token id and the
# decoded word at xb[5, 2]. Print both. (Use id2word to decode.)
# your code here
#
# Q6: What word is at xb[5, 2]?
#
# A6: (write the exact word here)

# TODO 2: Confirm the next-token alignment between x and y. Write code that
# checks, for the WHOLE batch at once, whether xb[:, 1:] is exactly equal
# to yb[:, :-1] (every element, not just spot-checking). Print a single
# True/False using torch.equal.
# your code here
#
# Q7: Did torch.equal return True or False? Why must this particular
# relationship (xb shifted by one equals yb) always hold by construction —
# point to the exact line in TokenWindowDataset.__getitem__ that guarantees
# it.
#
# A7: (True/False, plus the line you're pointing to)


# =====================================================================
# EXERCISE 2: What Embedding Vectors Actually Encode
# =====================================================================
# Training real embeddings is out of scope here, so instead we'll hand-craft
# a tiny, deliberately transparent embedding table where every vector is
# built from interpretable "concept" dimensions: [royal, male, female, young].
# This lets every answer below be checked exactly, not "approximately."

# --- Provided code: a. the embedding table ---
vocab2 = ["king", "queen", "man", "woman", "prince", "princess", "boy", "girl"]
hand_vectors = {
    "king":     [1, 1, 0, 0],
    "queen":    [1, 0, 1, 0],
    "man":      [0, 1, 0, 0],
    "woman":    [0, 0, 1, 0],
    "prince":   [1, 1, 0, 1],
    "princess": [1, 0, 1, 1],
    "boy":      [0, 1, 0, 1],
    "girl":     [0, 0, 1, 1],
}
w2i2 = {w: i for i, w in enumerate(vocab2)}
E = torch.tensor([hand_vectors[w] for w in vocab2], dtype=torch.float32)
print("E.shape:", E.shape)

# Q1: Looking at E.shape, how many words are in this toy vocabulary, and how
# many concept dimensions does each embedding vector have?
#
# A1: (write the two exact integers here, as "words=_, dims=_")

# --- Provided code: b. nearest neighbor by raw dot product ---
def nearest_by_dot(word, exclude_self=True):
    v = E[w2i2[word]]
    dots = E @ v
    order = torch.argsort(dots, descending=True)
    return [(vocab2[i], dots[i].item()) for i in order if not (exclude_self and vocab2[i] == word)]

print("nearest to 'king' by dot product:", nearest_by_dot("king"))

# Q2: Excluding "king" itself, which word is its nearest neighbor by raw
# dot product, and what is the exact dot-product value?
#
# A2: (write the word and the exact number here)

# TODO 1: Call nearest_by_dot("queen") and print the result.
# your code here
#
# Q3: Excluding "queen" itself, which word is its nearest neighbor, and
# what is the exact dot-product value?
#
# A3: (write the word and the exact number here)

# --- Provided code: c. a tie that raw dot product can't break ---
king_idx = w2i2["king"]
dot_king_king = (E[king_idx] @ E[king_idx]).item()
dot_king_prince = (E[king_idx] @ E[w2i2["prince"]]).item()
print("dot(king, king):", dot_king_king)
print("dot(king, prince):", dot_king_prince)

# Q4: Are dot_king_king and dot_king_prince equal? (True/False) Using the
# vectors for "king" = [1,1,0,0] and "prince" = [1,1,0,1], show the
# arithmetic (as a comment) for why "prince" picks up the same dot product
# with "king" as "king" gets with itself.
#
# A4: (True/False, plus the one-line arithmetic)

# TODO 2: Raw dot product can't tell "exact match" apart from "superset
# match." Cosine similarity can, because cos(v, v) is always exactly 1.0,
# the maximum possible value, for any nonzero v. Write a function
# nearest_by_cosine(word, exclude_self=True) that mirrors nearest_by_dot
# but divides by the product of the two vectors' norms. Then call it for
# "king".
# your code here
#
# Q5: Using cosine similarity, what is cos(king, king) and what is
# cos(king, prince), each to 4 decimal places? Is the tie from Q4 still
# present under cosine similarity? (True/False)
#
# A5: (write both numbers and True/False here)

# --- Provided code: d. analogy arithmetic ---
result_kmw = E[w2i2["king"]] - E[w2i2["man"]] + E[w2i2["woman"]]
print("king - man + woman:", result_kmw)
print("equals queen exactly?", torch.equal(result_kmw, E[w2i2["queen"]]))

result_pbg = E[w2i2["prince"]] - E[w2i2["boy"]] + E[w2i2["girl"]]
print("prince - boy + girl:", result_pbg)
print("equals princess exactly?", torch.equal(result_pbg, E[w2i2["princess"]]))

# TODO 3: Compute boy - man + woman as a vector. Find which vocabulary
# word it equals EXACTLY (use torch.equal against each candidate, or loop
# over vocab2). Print the matching word and the True/False confirmation.
# your code here
#
# Q6: Which word does boy - man + woman equal exactly?
#
# A6: (write the exact word here)

# --- Provided code: e. temperature-based "reconstruction" ---
# Treat cosine similarity to "king" as logits, and sample from
# softmax(logits / temperature) instead of always taking the argmax. This
# is the same mechanism a language model uses to pick the next token.
norms = E.norm(dim=1)
cos_to_king = (E @ E[king_idx]) / (norms * norms[king_idx])

for T in [0.1, 0.5, 1.0]:
    probs = F.softmax(cos_to_king / T, dim=0)
    print(f"T={T}:", {vocab2[i]: round(probs[i].item(), 4) for i in range(len(vocab2))})

# Q7: Looking at the printed probabilities, as T increases from 0.1 to 1.0,
# does king's own probability go up or down? Report the exact value of
# king's probability at T=0.1 and at T=1.0 (4 decimal places each, copy
# from the printed output — no rounding by hand).
#
# A7: (T=0.1 value, T=1.0 value, up/down)

# TODO 4: With torch.manual_seed(0) and T=0.1, draw 5000 samples from
# softmax(cos_to_king / T) using torch.multinomial(probs, num_samples=5000,
# replacement=True). Count how many times each word was sampled with
# torch.bincount. Print each word with its empirical percentage.
# your code here
#
# Q8: Report the two most-sampled words and their empirical percentages
# (rounded to the nearest whole percent). Note: exact percentages can vary
# slightly by PyTorch version/platform — report what your run produced.
#
# A8: (write the two words and their percentages here)


# =====================================================================
# EXERCISE 3: Byte-Pair Encoding (BPE), From Scratch
# =====================================================================
# Real tokenizers (GPT-style, etc.) use BPE merges learned from a large
# corpus. Here we train one from scratch on a tiny corpus so every merge
# step is small enough to inspect by hand.

# --- Provided code: a. the merge algorithm ---
def get_pair_counts(tokens):
    pairs = Counter()
    for i in range(len(tokens) - 1):
        pairs[(tokens[i], tokens[i + 1])] += 1
    return pairs

def merge_pair(tokens, pair):
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            new_tokens.append(pair[0] + pair[1])
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens

def train_bpe(text, num_merges):
    """Tie-break rule: among pairs with the highest count, pick the one
    that's lexicographically LARGEST as a tuple. This makes every merge
    step fully deterministic, with no dependence on dict ordering."""
    tokens = list(text)
    vocab = set(tokens)
    merges = []
    for _ in range(num_merges):
        pair_counts = get_pair_counts(tokens)
        if not pair_counts:
            break
        best_pair = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        best_count = pair_counts[best_pair]
        tokens = merge_pair(tokens, best_pair)
        merged_token = best_pair[0] + best_pair[1]
        vocab.add(merged_token)
        merges.append((best_pair, merged_token, best_count))
    return tokens, vocab, merges

# Q1: Before running anything, the initial (pre-merge) vocabulary is just
# the set of unique characters in `corpus` (defined in Exercise 1). Write
# code to print len(set(corpus)). What is the exact count?
# your code here
#
# A1: (write the exact integer here)

# TODO 1: Call get_pair_counts on list(corpus) DIRECTLY (before any merges)
# and find the single highest count value. Then find every pair that has
# that exact count — there should be more than one.
# your code here
#
# Q2: List every pair tied for the highest count at this very first step,
# along with that count value.
#
# A2: (write the pairs and the count here)

# --- Provided code: b. train and inspect ---
NUM_MERGES = 20
final_tokens, bpe_vocab, merges = train_bpe(corpus, NUM_MERGES)

print("final vocab size:", len(bpe_vocab))
print("number of merges actually performed:", len(merges))
print("first merge:", merges[0])

# Q3: Of the pairs you found tied in Q2, which one did train_bpe actually
# pick for merge #1 (check merges[0]), and which exact rule from the
# train_bpe docstring explains why that one won the tie?
#
# A3: (write the winning pair and the one-line rule)

# Q4: len(bpe_vocab) should equal the initial character count (your answer
# to Q1) plus len(merges). Verify this with code (print a single True/False
# from an equality check), then state the exact final vocab size.
#
# A4: (True/False, plus the exact final vocab size)

# TODO 2: Find the longest single token (by character count) anywhere in
# bpe_vocab. Print it and its length.
# your code here
#
# Q5: What is the longest token in the trained vocabulary, and how many
# original characters does that one token represent?
#
# A5: (write the exact token string and its length)

# --- Provided code: c. dump the full token dictionary ---
bpe_token2id = {tok: i for i, tok in enumerate(sorted(bpe_vocab, key=lambda t: (-len(t), t)))}
print("\nFull learned vocabulary (token -> id), longest tokens first:")
for tok, i in bpe_token2id.items():
    print(f"  {i:3d}: {tok!r}")

# TODO 3: Write an encode(text, merges) function that starts from
# list(text) and applies the merges in the SAME ORDER they were learned
# (loop over `merges`, calling merge_pair each time). Then use it to encode
# this held-out sentence, which was NOT part of the training corpus:
held_out = "Take one down, pass it around, 95 bottles of beer on the wall."
# your code here (define encode, then call it on held_out and print the result)

# Q6: How many tokens does `held_out` encode into? Compare that to
# len(held_out) (the raw character count) and report the compression
# ratio = len(held_out) / num_tokens, rounded to 2 decimal places.
#
# A6: (write the exact token count and the exact compression ratio)
