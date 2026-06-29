# LLM Implementation

## Reading

Raschke, Chapter 4

## Problems

1. This question pertains to the design of TransformerBlock, and its immediate surroundings in the overall GPTModel.  (See Figs 4.13, 4.15 and Listing 4.6).

   **a.** Why does it make architectural sense for the multihead attention module, and the feed forward module, both to have no final activation layer?

   **b.** Raschka charaacterizes `out_proj` in MultiHeadAttention as "optional". But, what essential problem with the attention head's function in the larger TransformerBlock would arise if we omit `out_proj`?  Consider the structure of the input to `out_proj` and the use of the MultiHeadAttention block's output in TransformerBlock.

   **c.** Must the shape of the input tensor to a TransformerBlock be the same as the output tensor, or might there be useful variations e.g. in the embedding size between sequential TransformerBlocks?  There is a clear answer to this question due to the design of the TransformerBlock

   **d.** Develop an equation for the number of trainable weights in a single, entire TransformerBlock, based on the relevant variables in __init__.  Use only variables that matter.  For the dimensions of GPT-2 (p 95) what is the trainable weight count (should be just over 7.08 million)?

   **e.** In Listing 4.6, we have two different LayerNorm modules, but a single Dropout module serves for both dropout actions in `forward`.  What do you think is the difference between these two types of module that demands two distinct LayerNorms?

2. Extend the work from problem 1 to arrive at an equation for the total trainable weights in the entire GPTModel, referencing Listing 4.7 and the table on p 95

   **a.** What are the trainable weights in the embedding layer and in the linear output layer, combined.  (Value a bit over 77 million)

   **b.** What about the positional embedding layer? (A little over 3/4 million.)

   **c.** And the TransfomerBlocks, in their entirety? (over 85 million)

   **d.** And the final LayerNorm?  Minor but part of the deal.

   **e.** Calculate the grand total, and validate against the bottom of p120.


3. In Listing 4.8, we use `argmax` to choose a token.  But if we instead chose according to the full softmax distribution, what might happen?  For a given input to `out_head`, for instance, would there be some tokens with a zero chance of being chosen?


4. In Listing 4.8, if we omitted the line `logits = logits[:, -1, :]` 
 
   **a.** And what would be the shape of `probas` and `idx_next` in this case?


   **b.** Considering input tensor `idx`, what element of it is likely to match idx_next[0, -2] in this case?


5. The text describes GPT-2 as having "tied weights" for `tok_emb` and `out_head`.  To understand this, we first have to imagine `tok_emb` not as a lookup table of embeddings but as a Linear layer, thus: `tok_emb = nn.Linear(cfg[vocab_size"], cfg["emb_dim"])` with 50257 inputs and 768 outputs.  It is fed a "one-hot" input designating the input token.  E.g. for token 1, it would get input [0, 1, 0, 0, 0, .....]

   **a.** In this view, where would the embedding for token 1 appear?  What weights of `tok_emb` would have that embedding?


   **b.** Why would a bias be inappropriate in this case?


6. Now imagine `out_head` as the same as the `tok_emb` we just discussed, but turned "upside down".  (Consult the line creating `out_head` on p. 119).  All the weights are identical to those for `tok_emb`.

   **a.** Say we input to `out_head` a perfect match for one of the token embeddings.  Is it guaranteed to choose that token?  Why or why not?


   **b.** It is documented that learned embeddings for common terms have higher magnitude (vector length) than less common terms.  So, for instance, the magnitude of "cat" may exceed that of "feline", even if both point generally in the same direction in embedding space.  What impact might this have if we present the embedding for "feline" to `out_head`?

