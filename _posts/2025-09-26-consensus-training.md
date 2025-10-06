---
layout: post
title: "Consensus Training: An Experiment in Generalization"
subtitle: "What if we only take gradient steps that everyone agrees on?"
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/path.jpg
tags: [ machine-learning, research]
---

### TL;DR

I experimented with a training method I call **Consensus Training**. The core idea is to only update model weights where the gradients from all examples in a mini-batch "agree" on the direction of the update (i.e., all positive or all negative). The hypothesis was that these "consensus" updates would represent more fundamental patterns and lead to better generalization, avoiding the learning of spurious cues specific to individual examples.

I tested this on a small algorithmic task from the Abstraction and Reasoning Corpus (ARC). The results were not what I'd hoped for. While the model did train with the consensus method, it showed no improvement in generalization compared to standard training. Based on this experiment, the initial hypothesis was not supported.

---

### INTRODUCTION

In recent years, neural networks have demonstrated astounding capabilities, tackling everything from generating creative text and images to making scientific discoveries. Yet, for all their advancements, the fundamental principles of how we train them have remained largely unchanged. We give a model a set of examples—the trainset—and use backpropagation to adjust its weights, nudging it towards a lower loss.

Given a sufficiently large model and enough training time, we can often achieve a near-zero loss on the trainset, meaning the model predicts every training example perfectly. Ideally, this mastery of the training data would translate to similar performance on new, unseen examples. However, this is often not the case. The model might fail to generalize, a phenomenon that represents one of the most fundamental challenges in machine learning. Instead of learning the true underlying patterns, the model latches onto "spurious cues"—quirks or noise specific to the trainset—that lead to perfect training performance but poor results on a test set.

The community has developed a host of techniques to combat this, such as weight decay, dropout, and early stopping. The intuition behind many of these is to constrain the model to prevent it from fitting the training data *too* perfectly. The modern approach for large language models (LLMs) often seems to be "just add more data," training on such a massive and diverse dataset that spurious cues become less reliable than the actual underlying patterns.

But let's consider a different perspective. If we assume our model is expressive enough to represent a perfectly generalizing solution, then there must exist a set of weights for which both the training loss and the test loss are zero. The problem is that there are likely *many* other sets of weights where the training loss is zero but the test loss is high. Standard training doesn't distinguish between these "good" and "bad" zero-loss solutions.

This suggests that the path we take through the loss landscape during training matters. From a given starting point, backpropagation can carve many different paths to a zero-loss valley. Some of these paths lead to generalizing solutions, while others lead to spurious ones.

This leads to the central question of this post: Can we do more than just decrease the loss? Can we "choose our path wisely" to bias the training process towards a more general solution?

### A Better Way to Aggregate Gradients: Consensus Training

Standard training aggregates gradients from a mini-batch by simple averaging. A mini-batch of, say, 32 examples is processed, 32 individual gradients are calculated, and the model takes a single step in the direction of their average. While effective, this averaging can hide important disagreements between examples.

Let's make this concrete. Imagine a single weight `w` and a mini-batch with just three examples. After the forward pass, we get three different gradients for this weight: `g1 = 0.5`, `g2 = 0.6`, and `g3 = -0.8`.

The standard approach would average them: `(0.5 + 0.6 - 0.8) / 3 = 0.1`. The optimizer would then take a small step in the positive direction. However, this step actively *increases* the loss for the third example, which was "voting" for a step in the negative direction. By following the average, we are accepting an update that, by definition, does not improve the outcome for every single example in our mini-batch.

I propose that this conflict is a root cause of learning spurious cues. A step that improves the loss for some examples at the expense of others is not a step that generalizes across the entire mini-batch. If it doesn't even generalize across the small set of examples we're currently looking at, why would we expect it to generalize to unseen data?

To counteract this, I propose an alternative aggregation strategy I call **Consensus Training**. The idea is simple: we only update weights where all examples in the mini-batch "agree" on the direction of the update.

Using our previous example with gradients `g1 = 0.5`, `g2 = 0.6`, and `g3 = -0.8`, there is no consensus. The gradients point in different directions. Therefore, under this new rule, we would simply not update this weight at all for this step. Its gradient would be masked and set to zero.

Now, consider a different scenario where the gradients are `g1 = 0.5`, `g2 = 0.6`, and `g3 = 0.2`. Here, all examples agree: the weight should be increased. A consensus exists. But what should the magnitude of the update be? To be conservative, we take the update with the smallest magnitude, `0.2`, as our final gradient. This guarantees that we are taking a step that every example agrees is in the right direction, and we are not "overshooting" the update required by any single example.

The core hypothesis is this: by only taking steps that have unanimous support across the mini-batch, we ensure that each update is a "generalizing step" across that batch. The hope is that a path built from these consensus-based steps is more likely to lead to a final model that generalizes beyond the trainset.

### The Experiment

To test this hypothesis, I needed a task where generalization is difficult and can be clearly measured. The [Abstraction and Reasoning Corpus (ARC)](https://arc-prize.org/) felt like a perfect fit. ARC tasks are designed to be solvable by humans with just a few examples, but they are notoriously difficult for modern machine learning models, which often fail to grasp the underlying abstract rule.

I chose a single ARC task and used a simple Convolutional Neural Network (CNN) as the model. The setup was as follows:
- **Model**: A small CNN suitable for the grid-based nature of ARC tasks.
- **Training**: I trained two identical models from the same starting weights. One was trained using a standard Adam optimizer, and the other was trained using Adam with my consensus gradient modification.
- **Data**: The few training examples provided by the ARC task.
- **Evaluation**: Performance was measured on the held-out test examples for that task.

The implementation of the consensus backward pass looks like this:

```python
def consensus_backward(model, loss_fn, data, targets):
    """
    Performs a backward pass where only gradients with consensus are kept.
    """
    # Get per-example gradients
    per_example_grads = []
    for i in range(len(data)):
        model.zero_grad()
        output = model(data[i].unsqueeze(0))
        loss = loss_fn(output, targets[i].unsqueeze(0))
        loss.backward()
        grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
        per_example_grads.append(grads)

    # Check for consensus and build the final gradient
    final_grads = [torch.zeros_like(p) for p in model.parameters()]
    num_params = len(final_grads)
    
    # This part is simplified for clarity
    for param_idx in range(num_params):
        # Stack gradients for the current parameter
        example_grads = torch.stack([g[param_idx] for g in per_example_grads])
        
        # Check for consensus (all positive or all negative)
        is_positive = example_grads > 0
        is_negative = example_grads < 0
        
        # Create a mask for each weight in the parameter tensor
        all_positive_mask = is_positive.all(dim=0)
        all_negative_mask = is_negative.all(dim=0)
        
        # Apply positive consensus
        if all_positive_mask.any():
            # Take the minimum gradient for consensus weights
            min_positive_grads = torch.min(example_grads, dim=0).values
            final_grads[param_idx][all_positive_mask] = min_positive_grads[all_positive_mask]

        # Apply negative consensus
        if all_negative_mask.any():
            # Take the maximum (least negative) gradient
            max_negative_grads = torch.max(example_grads, dim=0).values
            final_grads[param_idx][all_negative_mask] = max_negative_grads[all_negative_mask]

    # Set the final gradients on the model
    for p, g in zip(model.parameters(), final_grads):
        p.grad = g
```

### Results: A Null Finding


Both models—standard and consensus—were able to train and decrease their loss on the training examples. The consensus model trained a bit slower, which was expected since many potential updates are discarded at each step. I monitored the "consensus ratio" (the proportion of weights updated at each step) and found it hovered between 20% and 40%, so the model was consistently finding updates to make.

However, when evaluated on the test set, **the consensus model performed no better than the standard model.** Both models managed to solve the training examples but failed to generalize correctly to the test examples.

The core hypothesis—that enforcing consensus would guide the model to a more general solution—was not supported by this experiment.


