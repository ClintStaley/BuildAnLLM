# Build An LLM Midterm

Name_________________________________________________

## 1 22pts Vocabulary
Give terms for each of the following definitions, spelling correctly.

AI that creates documents, pictures, etc rather than simply analyzing and classifying patterns

Form of AI in which the program learns automatically from examples.  

New term for what was called "neural networks" 

Kind of neural network that generates images or movies 

Nonlinear function that frequently follows a linear neural network layer  

Specific example of such a function that is built on Relu and gaussian distributions  

A connection that adds the result of a deep learning module to its input.  

One pass through the entire set of training data 

One set of input/output samples large enough to be a reasonble example on which to optimize 

Tokenizing method used by e.g. GPT-2  

Deep learning layer that starts with a zero-mean 1-variance output, with individual per-dimension adjustments for mean and variance.  



## 2. 30pts Word Embeddings and Tokenization
a. 8pts Give an example of two words, with embeddings $E_1, E_2$ for which you might reasonably expect $\frac{E_1\cdot E_2}{|E_1||E_2|} \approx 1$.  Without changing your first two words, give a third word for which $\frac{E_1\cdot E_3}{|E_1||E_3|} \approx -1$.
```
First two words:

Third Word:
```
b. 8pts Assume we have three embeddings $E_1, E_2, E_3$, and $E_1 \cdot E_2 > E_1 \cdot E_3$. Even so, $E_3$ might point more closely than $E_2$ in the same direction as $E_1$?  How?
```


```
c. 14pts If I run BPE as implemented in our homework on the text "abracadabra", with a merge count of 3, what will the token dictionary be, in token number order? (Break pair-ties by alphabetic order, as in the homework code.  You should get 8 tokens total)  
```


```

## 3. 23pts Pytorch
a. 7pts Here is an interactive Python session working on a Pytorch tensor "pts".  Show the output of the final two prints.  
```
print(pts)

[[ 1.  2.]
 [-1.  3.]
 [ 3.  4.]]

print(torch.sum(pts, dim=0))




print(pts - (torch.sum(pts, dim=-2) / 3)



```

b. 7pts How far apart in memory are the elements pts.T[0, 0] and pts.T[0, 2]?
```
```
## 4. 10pts Linear Layer Design

Draw a Linear layer that is equivalent to this matrix:

$\begin{bmatrix}1&2\\3&4\\5&6\end{bmatrix}$
```



```

## 5. 20pts MultiHeadAttention
Below is a truncated selection from MultiHeadAttention.forward.  Write a Torch tensor expression that gives the key value for the second head, in the first batch, third token.
```
class MultiHeadAttention(nn.Module):
    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        pickedKey = ____________________________________
        ....
```

## 6. 27pts Transformer Block
The code for TransformerBlock.__init__ is below, along with a config dict.
Write the number of trainable parameters next to each element, assuming the
config given.
```
cfg = {
    "vocab_size": 2000,    
    "context_length": 256, 
    "emb_dim": 50,         
    "n_heads": 5,          
    "n_layers": 4,         
    "drop_rate": 0.1,       
    "qkv_bias": False      
}

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(             # Params: 
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)                 # Params: 
        self.norm1 = LayerNorm(cfg["emb_dim"])     # Params: 
        self.norm2 = LayerNorm(cfg["emb_dim"])     # Params: 
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])  # Params: 
```
