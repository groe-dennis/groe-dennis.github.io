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

# Vocabulary
generalization as in generalize beyond the trainset
generalizaion as in generalize trainset and testset
concept: a generalizing step.

# INTRODUCTION

In recent times, neural networks have shown tremendous capacilites in all manner of subjects. Predicting blabla, math, blabla

Yet, the pricicples of trainign have stayed the same since the beginning of training networks. Given a set of examples -called the trainset- , a neural network is adjusted through backpropagation with the goal to decrease the loss on the trainset. Given that the underlying neural network is expressive enough and it is trained for long enough, the model will converge to a loss of 0. This means that all train examples are predicted perfectly. Ideally, this would also mean that the model can predict examples that are similar to the trainset examples equally well. However, this is not necessaryly the case and insetad the model shows limited generalization ability.

In some ways this is the most fundamental issue in machine learning. The model does not learn the underlying algorithm that would generalize well, but instead lears some kind of spurious cues that make it have a perfect performance on the train examples and yet poor performance on the test set. 

Commonly addressed with techniques such as weight decay, early stopping, finding flat minima or in the case of modern llms just training on a bunch of data such that spurious cues are less likely and a generalizing solution emerges

Cursiouly, given the assumption that the model is capable enough to express the generalizing solution, there must exist one set of weights for the network where the trainloss is 0 and the testloss is also 0, meaning the model generalizes perfectly. The issue than is that there are multiple other sets of weights where the trainloss is 0, but the testloss is greater than 0, so it does not generalize. So the issue is to in addition to decreasing the loss on the trainset, also lead the model to a area where the found solution generalized. 


So, given that we don't change the loss function, for a initial set of weights there are multiple paths that backpropagation can take. Some paths will lead to a solution that generalizes well, and some will lead to a solution that does not generlaize well. 
A path here is a step-by-step application of backpropagation, where each step is one backward pass on a mini-batch of the trainset.

In this work, we propose to reframe the goal of machine learning from simply decreaseing the loss on the trainset to 'decreasing the loss on the trainset and also choosing a path wisely'. So its a stricter forunlation and by doing so we hope to go down a path that leads to a solution that generalizes better. 

(Insert image that shows for two weights a loss landscape that has multiple 0 points.)

Well, easier said than done. In this work we propose to choose a better path by handling the aggregation of the individual gradients in a more sophisticated way. 
Standardly during training of neural networks, a mini-batch of train examples is fed throught the network and the gradients for each example are calculated. Then, the gradients are averaged between the indivual gradients and this is the step that is taken. While simple and effective to decrease the loss, I believe that there are better aggregation functions that in addition to decreasing the loss also choose paths that generalize better.

In this work, we propse to do so by not updating all the weights where the gradients of the mini-batch do not all align. I call this Consensus gradients. 

The idea is simple. Suppose we have a weight w which gets the individual gradients g1=1, g2=-2 g3=2. If we take the average of those gradients as the step (1), then the loss for example 2 will increase, as we will move in the positive direction but in order to decrease the loss we would need to move in the negative direction. As such, this step will not decrease the loss for all examples and instead only for a subset of the examples.

I propose that this is the root cause of spurious cues. Because if we take an update that only decreases the loss on a part of the trainset, then by definition this step does not lead to a next weight state that generalizes accross the trainset and thus also does not generalize beyond the trainset. I hypothesize that the continous update that decreases the loss on only a part of the training examples leads to the emergence of those spurious cues and thus might contribute to models not generalizing as hoped for.

To counteract this behavior, the proposed Consensus grad approach only udpates those weights where the gradients do all align. For instance if the gradients are g1=1, g2=-2 g3=2 as before, this weight is masked and not updated. If the gradients do all align in the same direction however, like g1=1, g2=2 g3=2, the most conservative update is taken, in this case g=1. 
Thus we make sure that the step we take decreases the loss on the whole trainset and does not increase the loss on a subset of the trainset. 
The idea is that then the train step itself is one that generalize across the trainset and thus at least has the potential to also generalize beyond the trainset.  

All in all we still decrease the train loss, but we try to choose a path that leads to a generalizing solution.

# Introduction/Explaination

many solutions (accuracy 100% or not equivalent but related loss 0), some spurious and some generlaizing. how can be bias to be more generalizing? (related is flat minima etc)


In this work we look at the path
So even though all converge to 0 the path matters as some paths lead to generalizing, and some to spurious solution. 
generalizing step concept

how can we bias the path to a generalizing one? 

if we take an update where training examples disagree, then the result after the update will be a nn state that does not generalize across the trainset, and if it does not generlaize accross the trainset it will also not generlaize beyond the trainset. 

so the idea is to take only those directions in loss space where all examples agree. (and take the most conservative one). as such this will be a direction that decreases the loss on the whole trainset at once (and does not increase the loss for some examples). so this direction should be one that generalizes across the trainset and thus at least has the chance to generalize beyond the trainset.

another perspective: consensus updates are highly restrictive in the sense that the more away eamples are the less likley they are 



suppose we had a nn that is already close to the generalizing solution

The idea is that when

* this all only kinda works with small trainsets, else consensus ration will probably be too low

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


# ...
Wait is 0 also implizitly an update? Like if we do a 0 update, then the weight does not change but the same is true for grads 1, -1. (which doesnt mean 'lets not update' but instead 'well seems like we randomly got 0 as update')


----------------------------------------------------------

# INTRODUCTION

Deep neural networks now solve a wild variety of tasks—code generation, math, vision, reasoning—yet the core training loop is still the same recipe from decades ago: sample a mini-batch, compute per-example gradients, average them, take a step. Given enough capacity and optimization, the model typically reaches (near) zero training loss. But zero training loss is not the goal; what we actually want is: among all parameter settings that perfectly fit the training set, end up in one that also behaves sensibly on unseen data.

This is the crux: there isn’t just one zero-loss solution. There is a whole region (often a messy, high‑dimensional collection) of parameter configurations that interpolate the training set. Some correspond to “generalizing mechanisms,” others to “spurious hacks” (memorized shortcuts, brittle correlations, etc.). So the problem is not only minimizing the loss, but also navigating a path through parameter space that biases us toward solutions whose behavior extends beyond the training examples.

## Path dependence

Training is a sequence of gradient aggregation decisions. A “path” is the ordered list of updates (θ₀ → θ₁ → … → θ_T). Two runs with the same architecture and data but different update dynamics (batching, learning rate schedule, noise) can land in different zero-loss basins—with different generalization behavior. Standard SGD (or Adam) treats the mini-batch gradient average as a sufficient statistic. But averaging hides disagreement: if half the examples want a weight to go up and half want it to go down, the mean may be small—or even zero—yet that doesn’t mean “everyone is happy.” It means “they are fighting and we picked a compromise.”

This observation motivates reframing: not just “reduce training loss,” but “reduce training loss via steps that are themselves consistent across examples.” In other words: treat the micro-structure of per-example gradient directions as a signal about whether an update is likely to move toward a parameter region that generalizes across the *entire* training set (a necessary—though not sufficient—condition for generalizing beyond it).

## Hypothesis intuition (informal)

If an update direction for a specific weight increases loss for any training example, then immediately after that step the model sits at parameters that do *not* simultaneously reduce loss for the whole set. Repeatedly taking such “partial-consensus” steps might accumulate representational biases that encode narrow or spurious patterns—because we allow progress that benefits subsets at the potential expense of others. By contrast, restricting ourselves to updates where *all* examples agree on the sign of change for a weight (consensus) could:

1. Filter out directions driven by idiosyncratic examples.
2. Implicitly slow or freeze weights entangled with conflicting signals, forcing the model to adjust through invariant pathways first.
3. Bias the trajectory toward flatter, more “globally acceptable” regions (speculative).

This is deliberately strict—and in large, heterogeneous datasets would probably zero out too much signal. But in small-data or few-shot regimes (where overfitting via memorization is acute) it might act as a useful inductive bias.

## A minimal example (1D toy)

Imagine a single scalar weight w and three training examples producing per-example gradients: g = [+1, -2, +2]. The average is (+1 -2 +2)/3 = +1/3, so standard training nudges w upward. But one example (the -2) *wanted* it to go down strongly. After the step:
- Two examples improve.
- One gets worse.

A consensus rule would say: “No uniform direction exists—skip this weight this step.” Conversely, if g = [+0.4, +1.2, +0.3], all agree on +, so we take the *most conservative* shared magnitude (e.g., min |g_i| = 0.3) rather than the mean. That ensures every example’s loss moves (locally) in a descending direction for that weight.

## Proposed mechanism: Consensus gradients (informal definition)

For each parameter tensor:
1. Compute per-example gradients within the mini-batch.
2. Check sign alignment across examples elementwise.
3. If all non-zero signs match, keep that entry; otherwise mask it (treat as zero).
4. For retained entries, use the minimum absolute gradient magnitude among examples (a conservative step size).
5. Apply the resulting sparse, sign-consensus gradient instead of the usual mean.

So two filters happen: (a) sign unanimity; (b) minimal magnitude selection (to avoid letting a single large gradient dominate). The update is therefore a subset projection of the standard step—never moving in a direction that any example objects to (under the sign-based criterion).

## Relationship to existing intuitions

- Flat minima / sharp minima: Consensus filtering might steer away from brittle directions where gradients fluctuate strongly across examples. Not guaranteed, but philosophically adjacent.
- Gradient clipping / robust aggregation: Instead of clipping extremes, we discard contested coordinates outright.
- Multi-objective optimization: Each example is a tiny objective; we approximate a direction in the (very) intersection of their descent cones (but only using sign, so it’s a coarse proxy).
- Implicit regularization: By refusing a large fraction of potential micro-updates, effective capacity utilization slows, possibly delaying overfitting.

## Limitations (acknowledged up front)

- Overly strict: In realistic large-scale tasks, exact sign consensus is rare; the usable signal might collapse toward zero.
- Ignores magnitude disagreements when signs agree (could still encode conflict).
- “All examples agree” does not imply test generalization; it only enforces intra-batch coherence.
- Might underfit by freezing important but ambiguous parameters early.
- Sensitive to batch size: Larger batches decrease chance of unanimity.

## Why explore anyway?

Because it’s a clean, easily testable intervention at the level of gradient aggregation that operationalizes a crisp question: Does enforcing per-weight sign agreement inside mini-batches bias training toward more general solutions—especially in small-data reasoning-like tasks (e.g., ARC) where spurious shortcuts are tempting?

In short: we keep the loss function, optimizer outer loop, and architecture unchanged—only swapping the reduction rule. If the path matters, maybe a stricter definition of “acceptable” micro-steps nudges us into different basins. The rest of this post documents an initial implementation and small-scale experiment. Spoiler: the early evidence does not support a clear generalization gain—but the exercise helps sharpen why naive consensus might be too restrictive, and suggests softer variants (probabilistic agreement thresholds, weighting by agreement fraction, etc.) for future trials.


-------------------------------------------------------------
# GEMINI

// ...existing code...
concept: a generalizing step.

# INTRODUCTION

In recent years, neural networks have demonstrated astounding capabilities, tackling everything from generating creative text and images to making scientific discoveries. Yet, for all their advancements, the fundamental principles of how we train them have remained largely unchanged. We give a model a set of examples—the trainset—and use backpropagation to adjust its weights, nudging it towards a lower loss.

Given a sufficiently large model and enough training time, we can often achieve a near-zero loss on the trainset, meaning the model predicts every training example perfectly. Ideally, this mastery of the training data would translate to similar performance on new, unseen examples. However, this is often not the case. The model might fail to generalize, a phenomenon that represents one of the most fundamental challenges in machine learning. Instead of learning the true underlying patterns, the model latches onto "spurious cues"—quirks or noise specific to the trainset—that lead to perfect training performance but poor results on a test set.

The community has developed a host of techniques to combat this, such as weight decay, dropout, and early stopping. The intuition behind many of these is to constrain the model to prevent it from fitting the training data *too* perfectly. The modern approach for large language models (LLMs) often seems to be "just add more data," training on such a massive and diverse dataset that spurious cues become less reliable than the actual underlying patterns.

But let's consider a different perspective. If we assume our model is expressive enough to represent a perfectly generalizing solution, then there must exist a set of weights for which both the training loss and the test loss are zero. The problem is that there are likely *many* other sets of weights where the training loss is zero but the test loss is high. Standard training doesn't distinguish between these "good" and "bad" zero-loss solutions.

This suggests that the path we take through the loss landscape during training matters. From a given starting point, backpropagation can carve many different paths to a zero-loss valley. Some of these paths lead to generalizing solutions, while others lead to spurious ones.

(Insert image that shows for two weights a loss landscape that has multiple 0 points.)

This leads to the central question of this post: Can we do more than just decrease the loss? Can we "choose our path wisely" to bias the training process towards a more general solution?

### A Better Way to Aggregate Gradients

Standard training aggregates gradients from a mini-batch by simple averaging. A mini-batch of, say, 32 examples is processed, 32 individual gradients are calculated, and the model takes a single step in the direction of their average. While effective, this averaging can hide important disagreements between examples.

Let's make this concrete. Imagine a single weight `w` and a mini-batch with just three examples. After the forward pass, we get three different gradients for this weight: `g1 = 0.5`, `g2 = 0.6`, and `g3 = -0.8`.

The standard approach would average them: `(0.5 + 0.6 - 0.8) / 3 = 0.1`. The optimizer would then take a small step in the positive direction. However, this step actively *increases* the loss for the third example, which was "voting" for a step in the negative direction. By following the average, we are accepting an update that, by definition, does not improve the outcome for every single example in our mini-batch.

I propose that this conflict is a root cause of learning spurious cues. A step that improves the loss for some examples at the expense of others is not a step that generalizes across the entire mini-batch. If it doesn't even generalize across the small set of examples we're currently looking at, why would we expect it to generalize to unseen data?

### Consensus Gradients

To counteract this, I propose an alternative aggregation strategy I call **Consensus Training**. The idea is simple: we only update weights where all examples in the mini-batch "agree" on the direction of the update.

Using our previous example with gradients `g1 = 0.5`, `g2 = 0.6`, and `g3 = -0.8`, there is no consensus. The gradients point in different directions. Therefore, under this new rule, we would simply not update this weight at all for this step. Its gradient would be masked and set to zero.

Now, consider a different scenario where the gradients are `g1 = 0.5`, `g2 = 0.6`, and `g3 = 0.2`. Here, all examples agree: the weight should be increased. A consensus exists. But what should the magnitude of the update be? To ensure the step is beneficial for all, we should be conservative. We take the update with the smallest magnitude, `0.2`, as our final gradient. This guarantees that we are taking a step that every example agrees is in the right direction, and we are not "overshooting" the update required by any single example.

The core hypothesis is this: by only taking steps that have unanimous support across the mini-batch, we ensure that each update is a "generalizing step" across that batch. The hope is that a path built from these consensus-based steps is more likely to lead to a final model that generalizes beyond the trainset. We are still decreasing the training loss, but we are doing so more cautiously, choosing a path we believe is of higher quality.

