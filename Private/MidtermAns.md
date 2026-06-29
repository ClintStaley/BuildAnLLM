# Build An LLM Midterm

Name_________________________________________________

## 1 70pts Vocabulary

**Generative AI**

**Machine Learning**

**Deep Learning**

**Diffusion Model**

**Activation function**

**GELU**

**Residual Connection**

**Epoch**

**Batch**

**Byte Pair Encoding (BPE)**

**Layer Normalization**

## 2. 30pts Word Embeddings and Tokenization
a. 8pts Give an example of two words $W_1, W_2$, with embeddings $E_1, E_2$ for which you might reasonably expect $\frac{E_1\cdot E_2}{|E_1||E_2|} \approx -1$.  Without changing your first two words, give a third $W_3$ for which $\frac{E_1\cdot E_2}{|E_1||E_2|} \approx -1$.

b. 8pts Assume we have three embeddings $E_1, E_2, E_3$, and $E_1 \cdot E_2 > E_1 \cdot E_3$. Even so, $E_3$ might point more closely than $E_2$ in the same direction as $E_1$?  Why?

c. 14pts If I run BPE as implemented in our homework on the text "abracadabra", with a merge count of 3, what will the token dictionary be, in token number order? (Break pair-ties by alphabetic order, as in the homework code.)  

**a b r c d ab abr abra**

## 3 14pts Linear Algebra 

Describe, geometrically, the linear transformation done by this matrix, including any stretching, flattening, or rotating it does.

$\begin{bmatrix}1&0&1\\0&1&0\\1&0&1\end{bmatrix}$

**Stretches x and z dimensions (6pts), flattening them into the x=z plane (8pts).  Y unchanged.

## 4. 22pts Pytorch
a. Here is an interactive Python session working on a Pytorch tensor "pts".  Show the output of the final two prints.  
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
[[ 0. -1.]     5pts for basic idea 1pt for proper nesting  4pts for wrong axis\
 [-2.  0.]\
 [ 2.  1.]]  9pts**


b. How far apart in memory are the elements pts.T[0, 0] and pts.T[0, 2]?

**4 numbers apart 6pts**


## x. Show matrix translation of a Linear layer

## Index of weight that gives key for second head... 


## 5. Attention Block
Use MultiHeadAttention

Identify where the key from such and such a token is

## 6. Transformer Block
Use TransformerBlock

## 6. LLM Design
Use GPTModel

Name tensor sizes as they pass through LLM or Attention Code

Equalities in the text embedding

