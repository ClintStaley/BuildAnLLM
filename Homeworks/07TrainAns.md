# LLM Training

## Reading

Raschke, Chapter 5

## Problems
1. On p 139, why does the flatten call on logits require arguments, while that on targets does not?  

**logits is 3-D and we flatten only the first two dimensions.  Targets has only 2 dims, which we flatten entirely.**

2. Crossentropy is formally defined between two probability distributions p and q of the same domain (same number of possible outcomes), for instance p = [.2, .3, .5] and q = [.3, .4, .3].  H(p,q), is the "crossentropy of q relative to p", and is the p-weighted sum of the negative logs of the corresponding q values. er.

   $H(p,q) = -\sum_i p_i ln(q_i)$

   For the example above, that's -(.2ln(.3) + .3ln(.4) + .5ln(.3)) = 1.118.

   There is a strong mathematical theory underlying crossentropy, but at a high level it expresses the degree to which distributions differ and is almost universally used in deep learning to measure how far apart the probability-output of a deep learning model is from the correct probability.  (In deep learning it's sometimes called "categorical cross entropy" or CCE because probability distributions usually represent categories, e.g. "cat/dog/bird")

   a. What is H(p,p)?  Use p alone for both the weighting factor and the ln arguments.  You should arrive at a value *less* than H(p, q), because there is less difference between p and itself than between p and q. **1.03** 

   b. And, what about H(p, r) with r = [.9, .08, .02]?  Would you expect a smaller or larger value than H(p, q)?  **Larger; 2.73.**

   The functional.cross_entropy call, e.g. at the bottom of p139, makes probability vector A by a softmax out of the logits passed as first parameter, and a probability vector B with all zeros except for of the index numbers passed as the second parameter, which have all have equal nonzero values.  It returns the cross entropy between A and B

   With all that, explain how the cross_entropy call implements the averaging of negative log-logits for selected logits, as described in the text.  Is it computing H(A, B) or H(B, A)?  How does crossentropy in this case become equal to the average negative log of the probabilities of the correct logits?

   **H(B, A) It's using the 1/N values in B as weights, and weighing the negative log-probabilities corresponding to those, each with 1/N weight**

3. You measure the loss/error value after each batch, and find that it jumps up and down rather than gradually descending. Assuming there's no bug in the implementation, what is the likely explanation and fix? **Batch sizes too small and thus not representative.  Increase batch size**

4. Consider train_loader and val_loader as configured on p. 143, and as originally coded on p39. 

a. How many tokens are essentially wasted -- completely left out of both the training and validation process?  (It will be useful to know that int() truncates, not rounds) **int(5145 * .9) = 4630, so 4360 tokens are in the training set and 515 are in the validation set.  Each batch uses 512 of these, and 4630%512 = 22, but the final batch does include one more token for the output, so 21 are ignored.  If you adjusted stride to 

batch analysis.  Wastage of batches in DataSet??  Overall data size of DataSet.  total data, shifting percentage, wastage.  Redesign stride not to waste. Number of in/out pairs available, etc.

6. train_model_simple -- pick most costly operations... What happens if we move zero_grad up?  

