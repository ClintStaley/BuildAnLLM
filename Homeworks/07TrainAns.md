# LLM Training

## Reading

Raschke, Chapter 5

## Problems

1. Crossentropy is formally defined between two probability distributions p and q of the same domain (same number of possible outcomes), for instance p = [.2, .3, .5] and q = [.3, .4, .3].  H(p,q), is the "crossentropy of q relative to p", and is the p-weighted sum of the negative logs of the corresponding q values. er.

   $H(p,q) = -\sum_i p_i ln(q_i)$

   For the example above, that's -(.2ln(.3) + .3ln(.4) + .5ln(.3)) = 1.118.

   There is a strong mathematical theory underlying crossentropy, but at a high level it expresses the degree to which distributions differ and is almost universally used in deep learning to measure how far apart the probability-output of a deep learning model is from the correct probability.  (In deep learning it's sometimes called "categorical cross entropy" or CCE because probability distributions usually represent categories, e.g. "cat/dog/bird")

   a. What is H(p,p)?  Use p alone for both the weighting factor and the ln arguments.  You should arrive at a value *less* than H(p, q), because there is less difference between p and itself than between p and q. **1.03** 

   b. And, what about H(p, r) with r = [.9, .08, .02]?  Would you expect a smaller or larger value than H(p, q)?  **Larger; 2.73.**

   The functional.cross_entropy call, e.g. at the bottom of p139, makes probability vector A by a softmax out of the logits passed as first parameter, and a probability vector B with all zeros except for of the index numbers passed as the second parameter, which have all have equal nonzero values.  It returns the cross entropy between A and B

   With all that, explain how the cross_entropy call implements the averaging of negative log-logits for selected logits, as described in the text.  Is it computing H(A, B) or H(B, A)?  How does crossentropy in this case become equal to the average negative log of the probabilities of the correct logits?

   **H(B, A) It's using the 1/N values in B as weights, and weighing the negative log-probabilities corresponding to those, each with 1/N weight**


2. Consider train_loader and val_loader as configured on p. 143, and as originally coded on p39. 

   a. How many tokens are wasted -- completely left out of both the training and validation process?  (It will be useful to know that int() truncates, not rounds) 

**int(5145 * .9) = 4630, so 4360 tokens are in the training set and 515 are in the validation set.  Each batch uses 512 of these, and 4630%512 = 22, but the final batch does include one more token for the output, so 21 are ignored in the training set and 2 in the validation for a total of 23 wasted.**  

   b. Adjust stride in train_loader just enough to increase the number of batches to 10.  How many tokens are left unused in the training data now?

   **A 457 stride uses 457*9+513 = 4626, leaving only 4 unused**

3. In calc_loss_batch, p 144, why does the flatten call on logits require arguments, while that on targets does not?  

**logits is 3-D and we flatten only the first two dimensions.  Targets has only 2 dims, which we flatten entirely.**

4. In `train_model_simple` p147 , which statement do you think is the most expensive in computation, and which is the next-most?  Why?

**The calc_loss_batch class and the loss.backwards() are the contenders.  Any reasonable justification for choosing one as first is fine, though in practice the backwards is more expensive.**

5. In `train_model_simple`, how might you arrange to get he loss/error value after every batch?  What if you found that it jumps up and down rather than gradually descending. What is the likely explanation and fix? **Set eval_freq to 1.  Batch sizes too small and thus not representative.  Increase batch size**

6. In 5.4, what temperature, if any, would cause completely deterministic behavior by the generator -- so that the same word would be produced by the same context every time?  Explain your reasoning mathematically.

**None would do so.  No matter how low the temperature, there will still be some probability distribution, and the multinomial call will have some chance of choosing a different token.**

7. Using an LLM, build a Python program that:

   * Accepts as a commandline argument a positive integer dimension D, and also an optional "-gpu" flag
   * Creates three Python tensors T2d, and T3d of dimension DxD, and DxDxD, with elements drawn from a normal distribution with mean 0 and variance 4.
   * Do these two operations: a multiplication of T3d by T2d, arriving at a DxDxD result "scores", and a softmax on the final dimension of scores. For both, record the wallclock running time in mS and report it.
   * Report the average maximum probability value across all DxD softmax distributions.

Run this program on your local machine first, and try various dimensions up to the point where it runs for more than 15 seconds.  How does the average max probability change with D?  How does the runtime change with D?

Run the program with a -gpu flag if you have a CUDA GPU usable by Pytorch.  How does the runtime change?

Copy the program onto the Blackwell machine, and run the highest dimension you had, both without and with -gpu.  How does the run time vary?


