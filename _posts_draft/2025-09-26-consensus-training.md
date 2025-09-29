---
layout: post
title: Consensus training of neural networks
subtitle: Trying to take gradient steps for only those weights that generalize
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/path.jpg
tags: [draft]
---

# Appetizer

Neural networks are trained to decrease the loss on a dataset via backpropogation. A common critisim to this approach is that this means that the model is only able to interpolate between data points in the trainset and unable to extrapolate and generalize beyond it (even though for modern LLMs this is debatable).

In this work, I propose to tweak the training process in such a way that instead of training the model to predict the average of the trainset, it is only trained those weight updates that generalize. 


# TLDR
Neural networks are trained by averaging the gradients of mini-batches. As such, the step in the loss-landscape that is the average direction of the mini-batch gradients. I propose the following hypothesis: If, for a weight w the individual gradients g1,g2,...gn of the minibatch do not all point to the same direction, this weight update is one that does not generalize. The reasoning behind that is that then, one example wants to push the weight to the + direcion and the other weight to the - direction. Thus, which ever direction we end up taking for this weight, for at least one example the update will increase the loss. So this update can not generalize accross the whole trainset. 
Instead, we propose to restrict the gradient step to only update those weights, where the direction of all individual gradient updates align in one direction. Thus if we update one of those weights in this direction, all train examples agree that this will be a good step to take, reducing the loss for all examples. As such, this update is one that generalizes accross the trainset. The hypothesis is that then it will also be more likely to be a step that generalizes beyond the trainset.

However, I did not find any evidence that this is the case.

# Introduction/Explaination

# Experiments and results
We use ARC-AGI. Only one task, as we try to use this method to generalize from little data. We use this specific task: 

Results are that

1. We show that with normal training we have training curves as expected. Interestingly, even with normal training, the model seems to generlaize quite well.
2. With consensus training we can also train smoothly. However, its does not generlaize more. 


# Further observations
Consensus goes from 0.4 - 0.2 so we always have updates to take. 
Consensus might enable higher weight decay?



# Conclusion & Further Ideas
Based on this data we reject the hypothesis. 


