# Quiz 2

Name___________________________________

Assume the following code for the questions below.

```
batch_size, num_heads, seq_len, head_dim = 2, 4, 8, 10

K = torch.randint(-5, 5, (batch_size, num_heads, seq_len, head_dim)).float()
Q = torch.randint(-5, 5, (batch_size, num_heads, seq_len, head_dim)).float()

K_t = K.transpose( ___, ___) 
scores = Q @ K_t         
```
**1. 4pts** Fill in the missing transpose parameters -- with *nonnegative values* -- so that the multiplication will work.

**Ans: ..transpose(2, 3)**

**2. 6pts**
Write code that prints the logit (note there is no softmax) value for the degree to which the third token in the sequence is "of interest to" or is "attended by" the fifth token, for the first batch and the second head.

**print(scores[0][1][4][2])**

**3. 5pts** What does this print: `print(scores.shape)`?

**Ans: (2, 4, 8, 8)**

**4. 5pts** How many individual multiplications are performed by `scores = Q @ K_t`?

**Ans: 2(4)(64)(10) = 5120**

**5. 4pts** How long will the fully concatenated result value of the implied multihead attention structure be?

**(4)(10) = 40**
