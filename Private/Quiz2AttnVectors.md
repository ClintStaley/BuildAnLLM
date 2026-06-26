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
```

```
**2. 6pts**
Write code that prints the logit (note there is no softmax) value for the degree to which the third token in the sequence is "of interest to" or is "attended by" the fifth token, for the first batch and the second head.
```

```
**3. 5pts** What does this print: `print(scores.shape)`?
```

```
**4. 5pts** How many individual multiplications are performed by `scores = Q @ K_t`?
```

```
**5. 4pts** How long will the fully concatenated result value of the implied multihead attention structure be?
```

```