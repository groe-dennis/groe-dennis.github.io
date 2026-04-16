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

Ok also the lenght of the individual vectors matter, should also be the similar else the resulting minima is super different.

# Advancing Neural Network Performance through Emergence-Promoting Initialization Scheme
Their idea: Make init such that 'emergence' is best promoted. 
Method: Early layers (first half): divide weights by $  \alpha^k  $ (makes them smaller → fewer active neurons).
Later layers (second half): multiply by $  \alpha^k  $ (makes them larger → more active neurons).
Why:
They define an emergence measure $  E  $ based on counting paths in a graph. The math proves $  E  $ is maximized precisely when early layers are less active and late layers are more active — exactly what the asymmetric scaling does. 

They model the network as a directed graph (neurons = nodes, weights = edges). Emergence E is a kind of structural nonlinearity that quantifies how many “cross-scale paths” exist from inactive early nodes to active later nodes. E is literally the total weighted number of paths that start inactive (low scale) and end active (high scale).

This is inspired by Homological algebra which studies how a whole system differs from the sum of its parts. 
Its like the first layers are a bottleneck that filter information. If then some information gets through it should lead to complex behavior. So many paths from non-active to active neurons. 
So basically were complex behavior from very sparse input? and thus this complex behavior could not be predicted by the sum of its parts...

# A Categorical Framework for Quantifying Emergent Effects in Network Topology
Emergence defined as phenomena present in complex systems not explainable by the sum of its parts.
Sudden emergence of capability in nn during massive training.

## Setup
To understand emergence two concepts are needed:
1. Interactions of components of system (like particals colliding)
2. Global properties of a system, such as temperature, that are a simplification of the underlying processes

Emergence is then
Φ(s1 ∨ s2) ̸= Φ(s1) ∨ Φ(s2)
Where Φ is the the mapping to the global property and ∨ is the interconnection between two subsystems. 
So the global property of the interconnected system is not equal to the interconnection of the global properties of two systems.
(example: Φ as smooth function, ∨ as taking average)
The difference between the two sides of the equation can be studied with the derivative of f. For general functors, this leads to a derived functor of homological algebra.
Paper has some more nice examples of this.

Now they get into math that I don't understand. In chapter 6 however, they transalte that to a computational measure. In pseudo code

function compute_emergence(G, H):
    total_emergence = 0
    
    # Step 1: Find all deleted nodes (the ignored parts)
    deleted_nodes = all nodes in G that are NOT in H
    
    # Step 2: For each deleted node, look at where it directly points into the visible part
    for each x in deleted_nodes:
        starters = immediate outgoing neighbors of x that ARE in H   # these are the "entry points" into H
        
        # Step 3: For every entry point, count how many paths it can take inside H
        for each starter s in starters:
            num_paths = count_all_paths_from(s, target_set=H, subgraph=induced subgraph on H)
            total_emergence += num_paths
    
    return total_emergence

SO basically we construct H which is the observation, everything else is "deleteded". For the random boolean networks of the paper, they construct H by letting the network come to a stable state and every neuron that fires less than 5% is considered "dead". So it measures how much the dead neurons are still influecing the observed neurons bascially.

So, if ignored parts can create effects in your visible world that you could never have predicted by only watching the visible neurons alone, E is high.

If however, whatever patterns you see in H are almost completely explained by H itself, then E is low.

If we i.e. measure behavior of ants, E is high as we see complex behavior in H that can not be explained by H alone. Instead, it is determined by unobserved "dead" inputs, naming phermone traces.


H is not fixed but can be any summary/subsystem etc of the whole system. For nn, H could be the output layer of a network or all neurons that activate above some threshold in your data. 

Thinking further, my interpretation:
We can choose H arbitrarly, but depending on how we choose it, we can learn some more or less interesting things from it.
If we choose H to just be a small part of system, then (If H is almost detached from the whole system) then E will be trivially large, as there are many input paths simply because there are many dead variables.

If however, we choose H as a compressed version of system, then its interesting because
* If E is low, we would bascially be able to delete all the dead stuff and the bevior of the system would stay the same
* If E is high, deleting the dead stuff would destroy the emergent behavior in H, so the dead stuff is highly relevant.

With the ant example: If we observe the behavior of ants, if we were to disrupt their phermones, we would not see the emergent structure of foraging anymore. In that case, it would be easier to dessribe their movement by observation only(?)

Other E measure from different paper: E = Iout / Iin, so ratio of output information to input information.

They however also show that their E correlates with the Sparsitiy of the network. So the lower E is the less sparse the system also is. Though they argue that E is more refined in the sense that it needs sparsitiy at specifiy parts, not just overall sparsity.

# Quantifying Emergence in Neural Networks: Insights from Pruning and Training Dynamics
Very similar, but they measure E as inactive in early layer to active in later layer.
Realtive Emeregence as E/model params

Higher E with larger model size. Trivially.

Higher RelE: Compared to models of similar size its better emergent behavior
## Idea
In recursive networks, we dont have to rely so much on the later layer thing which is weird. Instead, later layer and earlier layers are the same?








# Parameter Symmetry Breaking and Restoration
Neural networks can have high symmetry, then most neurons are identical and model quite compressed. In low symmtry, model is less compressed and thus richer hirachies are possible. 

They measure the L₂ distance between the flattened weight vectors of the two neurons after sorting by norm, so the input weights. If that the distance is low the symmetry is high. (they sort because its then easier to calculate and still an upper bound, ie if the sorted is high, then also unsorted will be high)

Grokking: sudden generalization is a symmetry restoration event.

In-context learning (ICL): treated as symmetry restoration across layers or time. Early layers break symmetry to extract features; later layers restore symmetry to reuse those features flexibly (like a “compressed program” that works in new contexts).

Hierarchical learning / representation formation: breaking creates diversity (low-level features); restoration creates abstraction (high-level concepts).

## Idea


# LI2: A FRAMEWORK ON DYNAMICS OF FEATURE EMERGENCE AND DELAYED GENERALIZATION
They propose that learning with wd has three phases: 
1. Lazy Learning
2. Independent Feature Learning
3. Interactive Feature Learning



# Intrinsic Task Symmetry Drives Generalization in Algorithmic Tasks
## Quotes
"This indicates that weight decay is not
a necessary condition for generalization. From our perspective, weight decay plays a supplementary role, primarily
stabilizing and simplifying representations during the geometric organization stage."


"Intrinsic symmetries impose algebraic constraints on representations. Once internalized, these constraints reduce the
effective degrees of freedom of the model’s solution space."
##
Also three phases of learning
1. Memorization (arbitrary lookup table)
2. Symmetry Acquisition (model starts to respect symmetry constraints, so hypothesis space is reduced)
3. Geometric Organization (optimization biases push model to a simple manifold that satisifies all symmetries)

To Stage 2: Symmetries are constraints on what the true rule can be. Once the network respects them, the only solutions left are the actual algorithmic ones (e.g., addition as a group operation) (not sure about this strong statement)

They also push for symmetry explicitly, like in addition:
sym_violation = kl_divergence(softmax(logits), softmax(logits_swapped)).mean()
so KL between logits of addition problem and communatively or associatively swapped problem.

They also try other regularizes, like i nuclear norm
regularization, which penalizes the sum of singular values
and promotes low effective rank; (ii) entropy regularization, which discourages diffuse, high-energy embeddings;
and (iii) Lipschitz regularization, which enforces local
smoothness in representation space.

## My Idea
Okay if we generalize this, this is bascially reducing the hypothesis space and then searching for any solution in this hypothesis space. Similar to how if we ask a model to write code for ARC, the hypothesis space is reduced to code and if we find a solution in code, the model is likely to generalize. 
Can we arbitratrly find symmetries in ARC tasks?

# Grokking: From Abstraction to Intelligence

## Concepts:
CTM (Coding Theorem Method, approximation for KC): You randomly generate tons of tiny programs Run them Count outputs
Then:
Patterns that appear often → simple
Patterns that rarely appear → complex

BDM (Block Decomposition Method): Method for scaling CTM to larger objects.
Split into small tiles
Measure each tile’s complexity
Also account for repetition

entropy-style “geometric complexity”: 
C_geo =1−H(D)
H(D) = entropy (randomness)
So:
high entropy → random → low structure
low entropy → concentrated → high structure

Singular Learning Theory: Basically broader minima. They say that the posterior is governed by Free Energy. Free energy = fit + complexity where complexity = λ ln n. So if sample size increases, this term is more minimized and thus less complex and thus flat minima is found. 

## Methods
They use CMS. This is a causal method to check if a head is doing work to differentiate two labels. (not just deletion because neural nets are reduntant? so other parts could take over the functionality)

CMS(h) = [ M_theta(y2 | s_tilde) - M_theta(y1 | s_tilde) ] <-- Patched Logits
- [ M_theta(y2 | s1) - M_theta(y1 | s1) ] <-- Base Logits


The base term (second part) measures how much the model already prefers $  y_2  $ over $  y_1  $ on the original $  \mathbf{s}_1  $ (usually a large negative number, because $  y_1  $ is correct).


The patched term (first part) measures the same preference after we have injected the activation of head $  h  $ from the other example. (so activation of head h on last token position is copied from other example)

If in the patched version it can still differentiate, that head is useless to predict the right label for s1, so CMS. If its high, the model can't predict nicely anymore, so head was used in prediction. 

## Results
Early training (step 1k): Almost every head across all 48 layers gives high CMS values (bright heatmap everywhere). The network is using every part in a chaotic, redundant way → brute-force memorization.
Grokking moment (around step 10k): Many heads (especially in the middle layers) suddenly drop to CMS ≈ 0. The network is turning those heads off because it found a simpler way.
After grokking (step 100k): Only a few heads in the early layers and late layers still have high CMS. The entire middle block is dark (CMS near zero). You can literally skip those middle layers and accuracy barely changes.

So basically same findings as always(?), grokking manifests as structural “degradation,” where the network spontaneously prunes to a minimal effective circuit by rendering redundant parameters inactive or near-identity.

Grokking from Representation Collapse
Here grokking is viewed as a topological phase transition in the embedding space: high-entropy disordered representations collapse into a low-dimensional (so not only weights collapse but also activations)


# Complex behavior from intrinsic motivation to occupy future action-state path space
Nature paper.
First they say that extrinsic reward modeling has a number of issues. They instead argue that the fundamental drive is to maximize occupancy of future action-state path space. Rewards (like food or energy) aren't the end goal—they're just tools that enable more movement and exploration. goal-directedness emerges rationally to ensure movement never ends.

Action-state path entropy turns out to be the unique measure that satisfies intuitive properties of future occupancy (additivity, positivity, smoothness).

The paper flips the script: movement and exploration are the goal; rewards (energy, survival) are just fuel to keep occupying more path space indefinitely.

"The usual exploration-exploitation
tradeoff, therefore, disappears: agents that seek to occupy
space “solve”this issue naturally because they care about rewards only
as a means to an end"


The authors prove that the only mathematical measure consistent with intuitive properties of occupancy is action-state path entropy:

* It must be additive over time steps (so you can break long paths into shorter ones). (so long paths can be broken into shorter ones without loss of meaning):
The occupancy of a full trajectory equals the occupancy of the prefix plus the expected occupancy of the remaining suffix, conditional on the prefix.

* It decreases as probability increases (rare/unexplored paths contribute more "occupancy value"). Rare branches are more valuable because they open up more new path space.

* It must be differentiable/smooth for optimization.

# Comparision to Free Energy Principle (active inference / EFE) and Empowerment (MPOW)

Empowerment (MPOW): Maximizes mutual information between actions and future states. It prefers unstable fixed points of the dynamics and collapses to low-variability policies (agent stays near a few states/actions).

FEP (active inference / EFE): Often reduces to standard reward maximization in fully observable settings and produces near-deterministic low-risk behavior (avoids variability).

# Math
dont understand it completly, but they say that agents want to maximize expected future occupancy. “Occupancy” is not probability mass—it is how much new/unexplored future trajectory space the agent can cover. They then define condtions and then show that the only function satisfying all those conditions is $C(p) = -k \ln p \quad (k > 0).$ 
For one step this is exactly shannon entropy $C^{(1)}_i = -k \sum_j p_{ij} \ln p_{ij}.$
Then then expand this to infinite steps, ending at their formula.

The authors introduce weights $  \alpha > 0  $ (how much the agent cares about choosing diverse actions) and $  \beta \geq 0  $ (how much it cares about diverse next states)

Thus they provide a formula for the intrinsic return of a trajectory.

# Link to Neuroscience

Foraging behavior in bacteria, plants, and animals is well-described by random walks and Lévy flights, which remain valid descriptions even in familiar environments.” MOP agents naturally produce heavy-tailed (power-law) step-length distributions because high $  \beta  $ (state entropy) favors occasional long excursions that open vast new path space. This matches real animal foraging (even inside known territories) without needing an explicit “search reward.”

Our theory … captures the variability of behavior by taking it as a principle.” Even after convergence, the optimal MOP policy $  \pi^*  $ remains stochastic (never collapses to deterministic like reward-max RL). This matches empirical observations: animals and humans show persistent trial-to-trial variability in cortex, motor cortex, and decision tasks—even in highly familiar, over-trained settings

In a modified cartpole (right half of arena has controllable noise $  \eta  $), MOP agents with $  \beta > 0  $ actively seek the noisy side and show a clear optimum noise level that maximizes time spent there. The paper states: “for $  \beta > 0  $ there is an optimal value of the noise $  \eta  $ that maximizes the fraction of time spent on the right side … which is a form of stochastic resonance.” This reproduces the classic “noisy TV” effect in humans/animals: moderate randomness is preferred because it inflates next-state entropy without collapsing paths

Follow up in https://arxiv.org/pdf/2601.10276



# Transformers Represent Belief State Geometry in their Residual Stream
The authors argue that the model isn’t just memorizing statistics—it is learning to maintain and update belief states about hidden patterns in the data. These belief states are probability distributions over possible “hidden states” of whatever process generated the text. Using ideas from a field called Computational Mechanics, they prove (theoretically) and show (experimentally) that these belief states are linearly represented in the transformer’s residual stream—even when the geometry of those beliefs is extremely complex and fractal-shaped.
In plain English: the transformer ends up carving a geometric map of “what it believes is going on behind the scenes” directly into its activations. This map contains information about the entire future of the sequence, not just the next token.

An optimal predictor doesn’t just guess the next token. It maintains a belief vector η (a point in a probability simplex) that represents “how likely each hidden state is, given everything I’ve seen so far.” Every new token updates this belief according to a simple Bayesian update rule:

## Findings
Belief states are linearly represented, even for complex fractal beliefs (they find W and b such that porjection recovers the low dimentsional hidden state, test on hold out data)

Sometimes the geometry is spread across layers — When multiple different beliefs lead to the same next-token prediction (“degeneracies”), the full geometry collapses in the last layer but is preserved if you look at the concatenated residual streams from several layers. The loss only cares about next-token prediction. If two different belief states η₁ and η₂ produce exactly the same next-token probability distribution, the model has no incentive to keep them distinct right before the unembedding layer. the belief state geometry is spread across multiple layers of the residual stream

the transformer’s residual stream encodes the full long-range causal structure of the hidden process, not just the immediate next token.

As a general theory they state:Any data-generating process can be described by a hidden Markov model (HMM).
The optimal way to predict the next token is to maintain the belief state (posterior over hidden states) and update it Bayesian-style after every observation.
The set of all possible belief states forms a geometric object called the mixed-state presentation (MSP)—sometimes a simple simplex, sometimes a fractal.
When you train a transformer (or any next-token predictor with residual connections) on data from that process, the theory predicts that the model will internally represent exactly this MSP geometry, linearly, in its residual stream (or distributed across layers when degeneracies exist).
This is architecture-agnostic: as long as you do next-token prediction and have a residual stream, the geometry emerges.
Real language, ARC/AGI tasks, board games, etc., are usually non-stationary and/or non-ergodic (the underlying “rules” can drift or have long-term dependencies that never repeat exactly). In those cases the MSP can be infinite-dimensional or extremely high-dimensional, but the same mathematical object still exists and the theory predicts the same kind of belief-state geometry should appear (just harder to visualize or measure). The authors explicitly say the framework “will naturally extend” to these settings, but they leave empirical validation for future work.
## Idea
Check the internal beliefs of a transformer trained on only ARC task, can we somehow influence it? Are there interal beliefs that correspond to the correct label?

# https://www.sethmorton.com/blog/what_you_attend_to_cannot_be_static
Sounds potentially interesting, energy functions at different scales. No code though?

# Automated Continual Learning (ACL)
To overcome catastrophic forgetting, enabling continual learning.

Train a self-referential neural network that meta-learns its own in-context continual learning algorithm. The network literally modifies its own weights while processing a long sequence of tasks (just like how in-context learning works in LLMs, but extended to many tasks in a row).

Self-Referential Weight Matrices (SRWMs)
These are the heart of the model. They replace standard Transformer attention layers.
The weight matrix updates itself autoregressively using a fast rank-1 (delta-rule) update:$$W_t = W_{t-1} + \sigma(\beta_t) (v_t - \bar{v}_t) \otimes \phi(k_t)$$

Meta-Training Objective (the “ACL loss”)
For a sequence of two tasks A → B, the loss is:$$-\Bigl[ \log p(y_A' \mid x_A'; W_A) + \log p(y_B' \mid x_B'; W_{A,B}) + \log p(y_A' \mid x_A'; W_{A,B}) \Bigr]$$
First term: learn Task A well.
Second term: forward transfer (Task A should help learn Task B).
Third term: backward transfer (preserve Task A after learning Task B—this is what kills forgetting).


the model is a self-referential neural network whose weights literally rewrite themselves as it sees new tasks.
An SRWM is a single weight matrix $  \mathbf{W}  $ that:

Reads an input $  \mathbf{x}_t  $,
Produces an output $  \mathbf{y}_t  $,
AND updates itself via a fast rank-1 (delta-rule) update in a single forward pass.
During meta-training the network sees thousands of random task sequences (shuffled labels from Omniglot + Mini-ImageNet + FC100). At test time you just feed the real continual stream; the SRWM automatically does the right thing.

# The Geometric Inductive Bias of Grokking: Bypassing Phase Transitions via Architectural Topology
This paper asks: Is grokking caused by the Transformer’s architecture having too much freedom?
They use architecutral interventions to make grokking happen faster or to not make it happen at all.

## The two architectural interventions (the core of the paper)
The standard Transformer has two “extra” freedoms that let it prefer memorization-heavy solutions:

* Unbounded magnitude in the residual stream
Vectors can grow arbitrarily large → the network can encode information in size rather than angle.

* Data-dependent attention routing
The model can learn to attend to specific tokens differently for different inputs, breaking permutation symmetry.

## Intervention
A: Force all vecotors in the residual stream to be normalized. 
Result: The model is forced to use angular (direction-only) representations, which perfectly match the circular Fourier geometry of modular addition. Vector * unembedding becomes a cosine similarity.
→ Grokking is almost eliminated. Generalization happens in ~2,000–3,000 epochs instead of ~50,000+.


B: Intervention B: “Uniform Attention Ablation”

Override the learned attention scores and force every attention head to output a uniform distribution over tokens (e.g., [1/3, 1/3, 1/3] for a three-token input).
This turns the attention layer into a simple Continuous Bag-of-Words (CBOW) aggregator—completely data-independent and permutation-invariant.

Result: Even with ordinary LayerNorm, the model generalizes immediately (100 % test accuracy on every random seed) and completely bypasses the memorization phase.





# DISCOVERING GROUP STRUCTURES VIA UNITARY REPRESENTATION LEARNING
They state that groups are fundamental building blocks in many areas of mathematics and physics, yet indentifing groups from data is a challenge 

## Group
A group (G, ◦) is a set G with a binary operation ◦ that satisfies four axioms: Closure:
∀a, b ∈ G, a ◦ b ∈ G. Associativity: (a ◦ b) ◦ c = a ◦ (b ◦ c). Identity: There exists an identity
element e ∈ G such that for all g ∈ G, g ◦ e = e ◦ g = g. Inverse: For

## Method
It builds a model that is biased toward discovering structure, not just memorizing data. 

The authors design a learning system where:

Each symbol (A, B, C, …) is represented internally as a matrix
Combining two symbols (A ⋆ B) is done by multiplying their matrices

Why this matters:

Matrix multiplication is associative
Groups are defined largely by associativity + structure

So instead of checking if the learned rule is a group, the model is built so that group-like behavior naturally emerges.

“Build a system where only structured (group-like) solutions are easy to learn.”

-> So basically just T_hat = (1 / n) * trace(A[a] @ B[b] @ C[c]), map number to Matrix then do Matrix multiplication to get prediction. As Matrix multiplaction is associative, the whole algorithm will be associative as well. They also add regularization to make the Matrix well behaved. 
Idk seems not very interesting.


# Deep Learning is Not So Mysterious or Different