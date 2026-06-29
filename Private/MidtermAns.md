# Build An LLM Midterm

Name_________________________________________________

## 1 22pts Vocabulary

AI that creates documents, pictures, etc rather than simply analyzing and classifying patterns
**Generative AI**

Form of AI in which the program learns automatically from examples.  **Machine Learning**

New term for what was called "neural networks"  **Deep Learning**

Kind of neural network that generates images or movies  **Diffusion Model**

Nonlinear function that frequently follows a linear neural network layer  **Activation function**

Specific example of such a function that is built on Relu and gaussian distributions  **GELU**

A connection that adds the result of a deep learning module to its input.  **Residual Connection**

One pass through the entire set of training data **Epoch**

One set of input/output samples large enough to be a reasonble example on which to optimize **Batch**

Tokenizing method used by e.g. GPT-2  **Byte Pair Encoding (BPE)**

Deep learning layer that starts with a zero-mean 1-variance output, with individual per-dimension adjustments for mean and variance.  **Layer Normalization**

## 2. 30pts Word Embeddings and Tokenization
a. 8pts Give an example of two words $W_1, W_2$, with embeddings $E_1, E_2$ for which you might reasonably expect $\frac{E_1\cdot E_2}{|E_1||E_2|} \approx 1$.  Without changing your first two words, give a third $W_3$ for which $\frac{E_1\cdot E_3}{|E_1||E_3|} \approx -1$.

**Choice of synonyms for first two 4pts; antonyms for second 2 4pts**

b. 8pts Assume we have three embeddings $E_1, E_2, E_3$, and $E_1 \cdot E_2 > E_1 \cdot E_3$. Even so, $E_3$ might point more closely than $E_2$ in the same direction as $E_1$?  How?

**Ans: $|E_2| >> |E_3| can cause this 8pts**

c. 14pts If I run BPE as implemented in our homework on the text "abracadabra", with a merge count of 3, what will the token dictionary be, in token number order? (Break pair-ties by alphabetic order, as in the homework code.  You should get 8 tokens total)  

**a b r c d (5pts) ab (3pts) abr (3pts) abra (3pts)**

## 3 12pts Linear Algebra 

Describe, geometrically, the linear transformation done by this matrix, including any stretching, flattening, or rotating it does.

$\begin{bmatrix}1&0&1\\0&1&0\\1&0&1\end{bmatrix}$

**Stretches x and z dimensions (6pts), flattening them into the x=z plane (6pts).  Y unchanged.**

## 4. 23pts Pytorch
a. 7pts Here is an interactive Python session working on a Pytorch tensor "pts".  Show the output of the final two prints.  
```
print(pts)

[[ 1.  2.]
 [-1.  3.]
 [ 3.  4.]]

print(torch.sum(pts, dim=0))




```
**Ans: [3, 9] 7pts**
```
print(pts - (torch.sum(pts, dim=-2) / 3)



```
**Ans: \
[[ 0. -1.]   
 [-2.  0.]\
 [ 2.  1.]]  3x2 with mods 5pts Fully right 4pts**

b. 7pts How far apart in memory are the elements pts.T[0, 0] and pts.T[0, 2]?

**4 numbers apart**

## 10pts Show matrix translation of a Linear layer

Draw a Linear layer that is equivalent to this matrix:

$\begin{bmatrix}1&2\\3&4\\5&6\end{bmatrix}$

**Ans: 3 in, 2 out (5pts), weights in 12, 34, 56 from each input (5pts)

## 5. MultiHeadAttention
Below is a truncated selection from MultiHeadAttention.forward.  Write a Torch tensor expression that gives the key value for the second head, in the first batch, third token.
```
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads 
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        ....

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        pickedKey = ____________________________________
```
**keys[0, 2, head_dim:2*head_dim]**

## 6. Transformer Block
Use TransformerBlock

## 6. LLM Design
Use GPTModel

Name tensor sizes as they pass through LLM or Attention Code

Equalities in the text embedding

