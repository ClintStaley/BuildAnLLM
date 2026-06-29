# LLM Implementation

## Reading

Raschke, Chapter 4

## Problems

### 1. 
This question pertain to the design of TransformerBlock, and its immediate surroundings in the overall GPTModel.  (See Figs 4.13, 4.15 and Listing 4.6).

   **a.** Why does it make architectural sense for the multihead attention module, and the feed forward module, both to have no final activation layer?

   **They feed into residual connections and thus need a full range of output values to be useful "adjustments"**

   **b.** Raschka charaacterizes `out_proj` in MultiHeadAttention as "optional". But, what essential problem with the attention head's function in the larger TransformerBlock would arise if we omit `out_proj`?  Consider the structure of the input to `out_proj` and the use of the MultiHeadAttention block's output in TransformerBlock.

   **The output is residually added to the embedding.  Each head value would be able to adjust only a specific set of embedding dimensions rather than impacting/adjusting the entire embedding**

   **c.** Must the shape of the input tensor to a TransformerBlock be the same as the output tensor, or might there be useful variations e.g. in the embedding size between sequential TransformerBlocks?  There is a clear answer to this question due to the design of the TransformerBlock

   **The residual adds in the TransformerBlock require same-sized in/out tensors to work**

   **d.** Develop an equation for the number of trainable weights in a single, entire TransformerBlock, based on the relevant variables in __init__.  Use only variables that matter.  For the dimensions of GPT-2 (p 95) what is the trainable weight count (should be just over 7.08 million)?

   **Ans: $W = 4E^2 + E + 2(2)E + 4E(E+1) + (4E+1)E = 12E^2 + 10E = 7,085,568$**

   **e.** In Listing 4.6, we have two different LayerNorm modules, but a single Dropout module serves for both dropout actions in `forward`.  What do you think is the difference between these two types of module that demands two distinct LayerNorms?

   **LayerNorm has trainable weights and thus we need distinct instances.  Dropout just does the dropout action without any trainable weights**

2. Extend the work from problem 1 to arrive at an equation for the total trainable weights in the entire GPTModel, referencing Listing 4.7 and the table on p 95

   **a.** What are the trainable weights in the embedding layer and in the linear output layer, combined.  (Value a bit over 77 million)

   **Ans: $2*vocab*embed = 77,194,752$**

   **b.** What about the positional embedding layer? (A little over 3/4 million.)

   **Ans: $seqLen*embed = 786,432$**

   **c.** And the TransfomerBlocks, in their entirety? (over 28 million)

   **Ans: $numLayers(W) = 85026816$**

   **d.** And the final LayerNorm?  Minor but part of the deal.

   **Ans: $embed*2 = 1536$**

   **e.** Calculate the grand total, and validate against the bottom of p120.

   **Ans: 106,303,488**

3. In Listing 4.8, we use `argmax` to choose a token.  But if we instead chose according to the full softmax distribution, what might happen?  For a given input to `out_head`, for instance, would there be some tokens with a zero chance of being chosen?

**Ans: No.  No logit would be -inf, so all tokens would have a chance**


4. In Listing 4.8, if we omitted the line `logits = logits[:, -1, :]` 

   **a.** What would be the shape of `probas`?

   **(batch, context_size, vocab_size)**
   
   **b.** And what would be the shape of `probas` and `idx_next` in this case?

   **(batch, context_size, embed_size) and (batch, context_size)**

   **c.** Considering input tensor `idx`, what element of it is likely to match idx_next[0, -2] in this case?

   **idx[0, -1] since we'd expect the second to last token in the output to match the last in the original input**

5. The text describes GPT-2 as having "tied weights" for `tok_emb` and `out_head`.  To understand this, we first have to imagine `tok_emb` not as a lookup table of embeddings but as a Linear layer, thus: `tok_emb = nn.Linear(cfg[vocab_size"], cfg["emb_dim"])` with 50257 inputs and 768 outputs.  It is fed a "one-hot" input designating the input token.  E.g. for token 1, it would get input [0, 1, 0, 0, 0, .....]

   **a.** In this view, where would the embedding for token 1 appear?  What weights of `tok_emb` would have that embedding?

   **The embedding would appear as the 768 weights fanning out from the second input**

   **b.** Why would a bias be inappropriate in this case?

   **Bias would simply add the same amount to every embedding coordinate -- no value**

6. Now imagine `out_head` as the same as the `tok_emb` we just discussed, but turned "upside down".  (Consult the line creating `out_head` on p. 119).  All the weights are identical to those for `tok_emb`.

   **a.** Say we input to `out_head` a perfect match for one of the token embeddings.  Is it guaranteed to choose that token?  Why or why not?

   **No.  Cosine will be a perfect match, but magnitude of other embeddings might compensate.  Other logits and probabilities could be higher**

   **b.** It is documented that learned embeddings for common terms have higher magnitude (vector length) than less common terms.  So, for instance, the magnitude of "cat" may exceed that of "feline", even if both point generally in the same direction in embedding space.  What impact might this have if we present the embedding for "feline" to `out_head`?

   **Could choose `cat` instead due to higher magnitude causing higher dot product.**
