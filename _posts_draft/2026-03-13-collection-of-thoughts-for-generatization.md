---
layout: post
title: 
subtitle: 
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/path.jpg
tags: [draft]
---

# https://arxiv.org/pdf/1812.10156
Maybe large models just have better function init? hypothesis, larger models have simpler functions thus generalize better, or at least this helps generalization

# Scaling laws
Are there scaling laws for arc agi individual task training? so just with the few train examples, does val loss go down with scale

# NeuralGrok
https://arxiv.org/pdf/2504.17243
NN that transforms gradient for better val acc, improves grokking
Absolute Weight Entropy and Absolute Gradient entropy - lower equals better compression, better that taking weight norm?

# Progress Measures for Grokking on Real-world Tasks
Paper that introduces the Absolute weight entropy H(W) = − ∑ wi∈W |wi | ln |wi |
AWE automatically penalizes diffuse, high-entropy weight distributions (many medium-sized weights = memorization / overfitting) and rewards sparse, concentrated ones (few strong weights + lots of near-zeros = simple algorithmic circuit = generalization).

# Late-Stage Generalization Collapse in Grokking: Detecting anti-grokking with WeightWatcher
They propose: typcial measures as proposed are not sufficient. they dont capture anti-grokking, which is sparse circuits but with super high magnitude I think. Instead propose Heavy-Tailed Self-Regularization (HTSR / SETOL) theory. The key scalar is α=2.

# FloE: On-the-Fly MoE Inference on Memory-constrained GPU
They compress network.
## My idea
What if we can represent a nn by a sum of RNG networks or if we can otherwise generate matrixes? then we could for each part forward just generate them and discard again, reducing size...

# Linear Transformers Are Secretly Fast Weight Programmers
wp is basicall if we imagine a linear transformer that has as input a b vs one that has as input only b, then the first one though the a basically has new weights
The same also happens in the brain, where short term the synaptic molecules are adjusted: Fast weight programming and linear transformers: from machine learning to neurobiology
Quadratic attention you cant model with a state that is continously updated

# Idea
Compress each weight matrix with an AE, then store only the decoder and the compressed values. then during forward pass, generate the weights on the fly and discard again. basically trading flops for memory.
Maybe also do that on the fly with KV Cache instead of offloading

# Explaining grokking through circuit efficiency
Two circuits in a model
C_mem, fast to learn, memorizes, needs large norm to fit many examples
C_gen, slow to learn, generalizes,produces confident (large) correct logits using far smaller parameter norm

With L2 norm, once the dataset size is large enough, the memprizing solution becomes more expensive, so the generalizing circuit starts to dominate

So I guess for small datasets,  there exists a low norm solution that memorizes all train data. Only for large datasets, there does not exist such a solution anymore

I suppose that the gen circuit needs more exploration to discover substructures, thus beeing slower to learn?
### Additional Info
They identify three minimal ingredients needed for grokking:

A generalizing solution exists (and works on test data).
The generalizing circuit is more efficient (smaller parameter norm for the same logit confidence/margin).
The generalizing circuit learns more slowly than the memorizing one.



# A Tale of Two Circuits: Grokking as Competition of Sparse and Dense Subnetworks
Similar to the last paper.
Competition between two subnetworks, a sparse one and a dense one
First the dense one dominates. then with more training, the sparse one, so only a few neurons get a lot of norm, so they dominate the output.
With weight decay, unused weights are gently pushed to 0 after fitting the trainset

Interesting: their generalizing solution is always 6 neurons. However, they say there exists a 4 neuron solution as well, but that one failes to get found by SGD

## Idea
Can we see a nn as a combination of many sparse subnetworks, and then one of those subnetworks gets selected?
can we make this more explicit, by acutally adding sparse subnetworks and maybe also using evolutionary approaches to select best ones?

# Grokking and Generalization Collapse: Insights from HTSR theory
For each layer weight matrix W, they:

Form the correlation matrix X = (1/N) WᵀW. (also wie korreliert eingang i mit eingang j ist, über alle output neuronen)

Compute its eigenvalues and empirical spectral density (ESD). (Ein Eigenwert λ λ sagt: "Wenn ich einen bestimmten Richtungsvektor (Eigenvektor) durch X X schicke, wird er um den Faktor λ λ gestreckt." Große Eigenwerte: Es gibt Richtungen, in denen die Gewichte sehr stark zusammenwirken (starke Korrelationen).)

Random-matrix theory says i.i.d. Gaussian weights give a Marchenko-Pastur (MP) bulk distribution.

Real trained weights deviate: the tail follows a power-law ρ(λ) ~ λ^(-α).



α ≳ 5–6 → random / underfit (no correlations)
2 ≲ α ≲ 5–6 → well-conditioned, good generalization
α ≈ 2 → theoretically optimal (universal target for generalization)
α < 2 (very heavy-tailed) → extreme correlations, overfitting risk

they find pre-grokking, grokking and anti-grokking (anti-grokking through correlation traps). 

## Idea
How does this measurement behave when we have binary/ternary networks?
Is there a perfect measurement?

# From Spikes to Heavy Tails: Unveiling the Spectral Evolution of Neural Networks
Modern deep neural networks often end up with heavy-tailed weight matrices during training. If you look at the singular values (or eigenvalues) of a weight matrix $  W  $, their distribution (called the empirical spectral density, or ESD) has a “fat tail” 
they go into why this emerages. First its a rank one spiky update, the next steps then diffuse this spike into the tail.
interesting: they have some argument why the gradient matrix is low rank.

then they also say why heavy tails are good for generalization: they just show correlation between alpha 2-2.5 not causation though

# Exploration vs Exploitation
In SGD, if we did GD with very low lr it would be mostly only exploitation. However, with larger lr, SGD or even noisly GD exploration gets higher.

# Mixup
You randomly pick two samples $  (x_i, y_i)  $ and $  (x_j, y_j)  $, then create a convex combination: $$\tilde{x} = \lambda x_i + (1 - \lambda) x_j$$ $$\tilde{y} = \lambda y_i + (1 - \lambda) y_j$$
## Idea
Can this be done in hidden space? Or somehow optimize such that the intermediate convex values are all valid?

# Xavier (Glorot) and Kaiming (He)
Init such that input and output variance are ca 1 (Kaiming for Relu)

# Towards Theoretically Inspired Neural Initialization Optimization
So given a init, they look at what would theoretically happen if each example would have its own optimization process. Where would it land? If they all land in the same area, that would indicate generalization. 
As they cant optimize everything indvidually they approximate with 
GradCosine (GC) — a brand-new, differentiable metric:
$$\text{GC} = \frac{1}{B^2} \sum_{i=1}^B \sum_{j=1}^B \frac{g_i \cdot g_j}{\|g_i\|_2 \|g_j\|_2}$$

Basically just taking one gradient step and checking how the cosine similarity of the individual gradients is. High GradCosine = gradients of different samples are almost parallel → optimization path is smooth and consistent. They prove mathematically that training loss + generalization error are upper-bounded by something directly related to GradCosine.

They use this insight to optimze a init, such that GC is maximized, via tiny learnable scalar multipliers $  \omega_k  $ to each layer’s weights:
$$\theta_M = \{\omega_k \cdot W_k\}$$

# Advancing Neural Network Performance through Emergence-Promoting Initialization Scheme
Their idea: Make init such that 'emergence' is best promoted. 
Method: Early layers (first half): divide weights by $  \alpha^k  $ (makes them smaller → fewer active neurons).
Later layers (second half): multiply by $  \alpha^k  $ (makes them larger → more active neurons).
Why:
They define an emergence measure $  E  $ based on counting paths in a graph. The math proves $  E  $ is maximized precisely when early layers are less active and late layers are more active — exactly what the asymmetric scaling does. 
E is basically 