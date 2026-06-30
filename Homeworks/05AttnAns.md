# Attention

## Reading
Raschke, Ch 3

## Problems

1. In the attention blocks discussed in the chapter, K, Q, and V all have the same `head_dim` value.  Must they?  If so, why?  If not, which of them could have a different dim and why?  (Don't worry about how the attention block will fit into a Transformer block at this point -- that's a chapter 4 issue)

**K and Q must have same dim since they are dot-producted, but V may have an independent dim from K,Q**

2. Write an equation taking the embedding d_in 'E', d_out 'N', number of heads 'H', sequence length 'S', and batch size 'B' of a MultiheadAttention block (Raschke Listing 3.5), and returning the number of trainable parameters/weights the block will have.

   a. Derive the needed equation, without LLM help.  Assume that bias for the kqv Linears is false but note that for nn.Linear the default bias is "true".  What values do you get for: (E, N, H, S, B) = (128, 256, 8, 100, 4) and for GPT-3's values of (12288, 12288, 96, 2048, 16).  As a check, you should get just over 164000 for the first, and just under 604 million for the second.

**$P = 3EN + N^2 + N; 164096, 603,992,064**

   b. Of the five variables specified in part a, three do not in fact affect parameter count.  Which three, and why?

**Sequence length and batch size, since the same computation applies to all tokens in the sequence, and across batches.  And head count, since ultimately we divide ExN matrices/linears into heads, but do not increase their size doing so.**

   c. In chapter 4, we'll find that the input and output tensor dimensions must be the same for the attention block to fit properly into the larger structure of a "transformer block" of which it is the central element.  But, a small adjustment to the design of MultiHeadAttention still permits differing `d_in` and `d_out` while ensuring that the returned `context_vec` has the same dimensions as the input tensor `x` parameter of `forward`.  Adjust just a small part of one line in the __init__ method to ensure this.  And, revise your equation from a to reflect this change.

**Ans: out_proj = nn.Linear(d_out, d_in)  P = 4EN + E**

   This type of adjustment is actually done in more advanced attention block designs that have arisen in the last 5 years or so.

   d. Given your new design from part c, create an equation for the number of multiplications needed by one run of `forward`.  Note that 4 of the 5 variables will now matter.  And assume for simplicity that the causal masking is omitted and if you know what a "KV cache" is, assume one is not in use.  

   Apply the equation to the two cases cited earlier: (E, N, H, S, B) = (128, 256, 8, 100, 4) and GPT-3's values of (12288, 12288, 96, 2048, 16). (You should get just over 52 million and almost 20 trillion, respectively.)

**Ans: $M = 4SBEN; 52,428,800; 1.98x10^{13}$**

