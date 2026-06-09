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

## 1 Lazy learning
The top layer quickly overfits to whatever random junk the hidden layer is outputting. The network looks like it's just memorizing.
But crucially, weight decay (a tiny bit of L2 regularization, denoted η) makes the back-propagated gradient $  G_F  $ from the top layer suddenly carry real information about the target labels. 
NOTE: We know that grokking also happens w/o weight decay, so maybe in theory this hold up, but its not soley due to wd

## 2 Independent Feature Learning

Last layer gives some information, the neurons in layer n-1 learn independently. For multi layer networks they say that first the first layer learns, then the second one, etc.
Each neuron ascends its own energy function. For their modulo addition, they converge to 'irreducible representations', some math things that minimally describes the modulo addition.
You only need roughly $  2(M-1) = 8  $ hidden neurons (Theorem 3) to perfectly reconstruct the target — versus $  M^2 = 25  $ neurons if you just memorized every possible pair.

Stage 2 rresponds to nonlinear canonical-correlation analysis (CCA) between the input X and target Y

## Interactive Feature Learning

Stage III kicks in after Stage II has produced some (but not necessarily all) irrep features. Now the hidden neurons are no longer independent (?)

Now, similar features get repulsed. Once two neurons $  j  $ and $  l  $ have similar activations $  \mathbf{f}_j \approx \mathbf{f}_l  $, the effective gradient matrix $  B  $ has a negative entry $  b_{jl} < 0  $. This pushes the two neurons away from each other so they specialize on different irreps, increasing diversity.

Suppose the current hidden representation only spans a subset $  S  $ of all irreps. Then, the gradient becomes such that automatically zeros out the gradient for already-learned irreps and boosts the gradient exactly on the missing irreps. 
→ The energy landscape is dynamically reshaped so that the remaining neurons are attracted only to the missing local maxima.


## Other
The paper emphasizes that residual connections  are extremely helpful here: they provide a cleaner, less-noisy gradient path, bypassing the random re-weighting that would otherwise scramble the signal in deep stacks 

Bonus: The recent Muon optimizer accelerates exactly this Stage III by suppressing gradients that would duplicate already-learned features, making the network explore missing irreps much faster.

Check https://grok.com/c/976d0adb-06c5-486d-ae3d-9edb61aa364b?rid=228e81bf-b57f-46da-b876-ff5bce334180 for some follow-up, grok was busy



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

To Stage 2: Symmetries are constraints on what the true rule can be. Once the network respects them, the only solutions left are the actual algorithmic ones (e.g., addition as a group operation)(Note the other paper on groups!) (not sure about this strong statement)

They also push for symmetry explicitly, like in addition:
sym_violation = kl_divergence(softmax(logits), softmax(logits_swapped)).mean()
so KL between logits of addition problem and communatively or associatively swapped problem.

They also try other regularizes, like i nuclear norm
regularization, which penalizes the sum of singular values
and promotes low effective rank; (ii) entropy regularization, which discourages diffuse, high-energy embeddings;
and (iii) Lipschitz regularization, which enforces local
smoothness in representation space.


They did not check if in-batch symmetry is needed, or whole dataset sufficies. (However for their KL they need it)

I guess one additional point they have is that as soon as symmetry loss is 0, generalization loss is also 0, afterwards they then have lower norm

weight decay is not a necessary condition for generalization. weight decay plays a supplementary role, primarily stabilizing and simplifying representations during the geometric organization stage.

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

The author’s core claim is that the three big “anomalous” generalization behaviors everyone talks about—benign overfitting, overparametrization, and double descent—are actually completely normal once you use the right theoretical lens.

People used to think:

“Deep nets have way more parameters than data points → they should overfit horribly.”
“Yet they generalize amazingly well.”
“They even fit random noise perfectly and still work on real data.”

This led to the narrative that “deep learning broke classical statistics and we need entirely new theory.”
Wilson says: No, we don’t. The same behaviors appear in linear models, polynomials, and Gaussian processes when you give them the right soft inductive bias. The mystery disappears if you stop using the wrong tools (VC dimension, Rademacher complexity) and start using the right ones (PAC-Bayes and countable hypothesis bounds).

## Key Idea 1

Traditional thinking: To avoid overfitting, you restrict the hypothesis space (e.g., force the model to be convolutional so it has built-in translation invariance).
Wilson’s alternative: Give the model an enormously flexible hypothesis space, but add a soft preference for simpler solutions. The model can fit anything, including noise, but it really likes low-complexity (compressible) solutions that are consistent with the data.
(so basically just regularization lol?)

## Key Idea 2
 Classical generalization theory (VC dimension, Rademacher complexity, fat-shattering dimension) only looks at how large your hypothesis class $  \mathcal{H}  $ is. These bounds get worse as you add parameters, so they cannot explain why bigger models generalize better. Wilson says the right tools are PAC-Bayes and countable hypothesis bounds. They shift the focus from the size of $  \mathcal{H}  $ to which solutions inside $  \mathcal{H}  $ your prior prefers.



## Explain phenomena

Larger models actually give you more opportunities to find very flat, simple solutions.

Double descent: First underfit. Then overfit. But then, we start to get many many 0 loss solutions, many of which are low complexity? IDk. but easier to find I guess.

Ok so basically make model large enough to find low complexity solutions. 

ICL: It selects the right “kernel” (or inductive principle) on the fly from the patterns it saw during pre-training — essentially acting like a mixture of Gaussian-process experts.

Larger models have a built-in compression bias. After training they often end up more compressible than smaller models (Maddox et al. 2020; Goldblum et al. 2024).

Effective dimensionality Neff = sum (eigenvalue_i/(eigenvalue_i+alpha))
It counts “how many directions really matter.”
Eigenvalues much larger than alpha contribute ~1; tiny eigenvalues contribute ~0.

Think of the loss landscape as a multi-dimensional valley. Each eigenvalue $  \lambda_i  $ tells you how steep the valley is in one particular direction (eigen-direction):

Large $  \lambda_i  $ (much bigger than $  \alpha  $): very steep wall.
Tiny change in that weight direction → huge increase in loss.
This direction is sensitive. You must store that weight with high precision; you cannot round it or quantize it much without breaking the model.
→ Contributes almost +1 to $  N_{\text{eff}}  $.

Small $  \lambda_i  $ (much smaller than $  \alpha  $): almost flat floor.
You can move the weight a lot in that direction and the loss barely changes.
This direction is insensitive. You can store that weight with very low precision (coarse quantization, pruning, etc.) and the model still works fine.
→ Contributes almost 0 to $  N_{\text{eff}}  $.


High $  N_{\text{eff}}  $ (e.g. = 4) → many steep directions → the solution is fragile and needs high-precision storage → less compressible (higher Kolmogorov complexity).
Low $  N_{\text{eff}}  $ (e.g. = 1) → only one steep direction, all the others are flat → the solution is robust and can be stored with very few bits → more compressible.

## Idea
Can we minimize the eigenvalues as loss?

## Other
Regularization (e.g., weight decay) is a special case of a Gaussian prior.


# Muon Optimizer
https://chatgpt.com/share/69e77e4b-dc90-8325-b63c-8886798a955a
* **Adam optimizer** adapts per-parameter using gradient statistics; Muon adapts over time via momentum decay
* Adam is geometry-aware (rescales directions); Muon is time-aware (damps motion)
* Adam is stable from the start; Muon is aggressive early and stabilizes later
* Adam typically needs a learning rate schedule; Muon has built-in decay
* Adam normalizes updates; Muon reduces momentum over time



# Parameter Symmetry Potentially Unifies Deep Learning Theory

## Intro
high-level, unifying hypothesis paper that argues one elegant idea—parameter symmetry (and its breaking/restoration during training)—can tie together a huge number of seemingly unrelated phenomena observed in modern neural networks.

Deep learning is full of hierarchical, phase-transition-like behaviors:

Training suddenly “clicks” and loss drops sharply.
Networks magically match the right level of complexity to the task (not too simple, not overfitting).
Representations become beautifully structured and hierarchical (e.g., early layers detect edges, later layers detect concepts).

->
he authors’ central claim is:
Symmetry breaking and restoration are the unifying physical mechanism behind all three hierarchies (learning dynamics, model complexity, and representation formation).

They treat neural networks like physical systems (think magnets, water freezing, or superconductivity), where symmetry is a fundamental organizing principle.

## Paramter Symmetry
A model $  f_\theta  $ has a symmetry under a group $  G  $ if transforming the parameters $  \theta  $ by any $  g \in G  $ leaves the output unchanged:

f_g_transformed(x) = f(x)

Common symmetries in networks include:
Sign-flip symmetry in ReLU / tanh layers
Translation symmetry in attention/softmax, etc.
Scaling/rotation symmetries in linear layers

Symmetries make the network effectively lower-dimensional (many parameters are redundant). Training can break these symmetries (neurons differentiate) or restore them (neurons become identical again).

## The Three Hierarchies the Paper Unifies
1. Learning Dynamics Hierarchy (temporal phases during training)
Training often shows abrupt jumps and plateaus. these jumps frequently coincide exactly with moments of symmetry breaking.
Example: Start with small random initialization → network is highly symmetric → loss is stuck → then symmetry breaks → sudden rapid learning.

2. Model Complexity Hierarchy 
Even massively overparameterized networks behave as if they have far fewer effective parameters.
The authors propose a space quantization conjecture: with weight decay, neuron weights are forced to live on a discrete grid separated by a distance proportional to the regularization strength. This bounds the number of active neurons, no matter how wide the layer is.
the functional complexity of models adapts to the target function –
this is exhibited in simplicity biases [34], compressive coding through the information bottleneck [67, 68],
and the “blessing of dimensionality” in overparameterized nets

3. Representation Formation Hierarchy
distinct spatial structures arise in the layers of neural networks,
with progressively deeper layers tending to encode increasingly abstract information – this is evidenced
in the structured representations such as neural collapse [54], hierarchical encoding of features [77], and
universal alignment of representations across models


-> so symmetry can change through time or through layers. eural
networks are found to break symmetries in early layers and restore symmetries in final layers 
-> paramter symmetry as a unifying mechanism for deep learning

"With the recent advances in how arbitrary parameter symmetries may be deliberately introduced or removed (Section 6), it is now possible to design symmetries matching practitioner intentions."

## 2 Paramter Symmetry in Deep Learning
* Neural network architectures have many symmetries, for instance the self-attention layer is (among other symmetries) permutation invariant. 
* But I think they say both that symmetry can come from architecture so it holds for all parameters and also only holding for specific parameters.
* "the number of groups induced by these symmetries often grows exponentially in the size of the model"
* Symmetry means part of the solution paramters are redundant, so its a form of overparametrization
* "double rotation symmetry causes the self-attention layers to have a low-rank bias"

## 3 Learning Dynamics is Symmetry to Symmetry
* One primary effect of symmetry on the loss landscape is
that it creates extended saddle points from which SGD or GD cannot escape (?)


* Dynamics Hypothesis: The learning dynamics of neural networks are dominated by jumps between symmetry groups, with parameters going from a larger to a smaller group (symmetry breaking) or from a smaller to a larger group (restoration)

* Models initialized with a small norm are approximatly symmetric ()
* With those small init models, whenevery symmetry breaks (so distance between normal paramters and group transformed exeedes a threshold), learning happens, else it plateaus

## 4 Symmetry Adaptively Limits Model Complexity

* Complexity Hypothesis: Symmetry adaptively controls the model’s capacity. The model converges to a symmetry class whose complexity matches the complexity of the target.

* loss function symmetry: We apply g to paramters, and loss does not change
* paramter symmetry: we apply g to paramters and paramaters do not change
* Lazy learning: weights move only a little bit, essentially its linear regression on the random features

-> Theorem
If the loss function has G-symmetry, and the initial θ is G-symmetric, there is a model with fewer paramters whos learnign dynamics are the same as θ


* Also they conjecutre that with enough regularization, the amount of different neurons in a layer is constant, even if width goes to infinity

## 5 Representation Learning Requires Parameter Symmetry

* earlier layers encode a large variety of low-level features and later layers learn a composed and abstract representation that is invariant to the changes in the low-level details
-> So if there are small changes such as a shifting of a cat, the later layers will not notice that

* If later/last layers collapse as many directions as there are classes, this corresponds to good generalization, if its more scattered its less good at generalizing

* they train a model with removed permutation symmetries (they do this with a method called syre: static (fixed) random Gaussian bias (theta+theta_bias), sampled once at the beginning, never changes Thus with weight decay on theta it aligns/onverges with theta_bias. Thus, unlike with normal wd which aligns towards 0, )
-> after doing syre, the innerclass variance does not vansih

* based on their experiments they propose 3 regimes in the layers of a trained nn
1. the first few layers of neural networks serve as an expansion phase where the representation becomes linearly separable (a straight line can seperate the features, or a linear layer can. raw input like images is typically not linearly seperatable), which requires the layer to be wide and implies a high rank. So They transform the input into a higher-dimensional or more expressive space where class distinctions become clearer. (Note: Kinda like vipassana) The expanded representation uses many nearly independent directions to "unfold" the manifold and separate classes. Low-rank structure would constrain the features too much, preventing the necessary expressivity for separation. Thus, early layers tend to show increasing or high representation rank
2. then, a “reduction” phase happens where the irrelevant information is thrown away and the neurons encode more and more compact information
3. lastly, a “transmission” phase where the layers do nothing except transmitting the signal it receives


(Three phases remind me of the emergence paper)

### Platonic representation hypothesis
Different neural networks, when trained well, tend to learn very similar internal representations of the data — even if they have different architectures, different random initializations, or are trained on slightly transformed versions of the data.

They propose that parameter symmetry, especially double rotation symmetry, is the key mechanism driving this universality.

* Two networks A and B have learned a universal representation if, for any two inputs x₁ and x₂, the following holds approximately: h^A₁(x₁) · h^A₂(x₂) ≈ h^B₁(x₁) · h^B₂(x₂)
-> so the dot product, so the similaritiy between two representations should be the same in two networks (this is an idealization, in practise, the degree of similarity is measured)

* When neural collapse happens, all examples of the same class collapse to a single point, and the class centers form a regular simplex (equally spaced). Because of this highly structured geometry, any two networks that both achieve NC must automatically have aligned representations. So NC is a strong form of universal representation.

* This all is special, because due to the double rotation symmetry, there exist infinitely many global minima for a deep linear network such that the representations are not aligned. Yet SGD finds the one that are aligned.
-> This is only possible if the first layer transforms the representation into
an input-independent form (The first layer of each network must learn to remove the specific quirks or transformations that are unique to its own input data.)
-> in the degenerate manifold of solutions, the training algorithm prefers a particular and universal one (SGD + stochastic noise (the “entropic force”) prefers the simplest / lowest-volume solutions on this manifold.)

### Double Rotation Symmetry
also sometimes called coupled or joint rotation symmetry
In Self-Attention:
* Rotation of Q and then simultaneously apply the inverse transpose rotation to K
-> the attention output remains exactly the same.
-> This creates a huge continuous family of equivalent solutions (a “degenerate manifold”).

## 6 Mechanism and Control
* stochastic dynamics tend to move to places that are “cold,” a common phenomenon in nature (so symmetric) (In fluids with a temperature gradient, particles (especially large molecules like DNA) tend to move toward the colder region. This happens even without obvious forces — it's a pure entropic/stochastic effect.) (noisy (Brownian) dynamics in systems with position-dependent temperature or noise strength lead to particle accumulation in the "colder" (lower noise / lower fluctuation) areas.)

* In deep learning → parameters move toward more symmetric regions because those regions have lower effective temperature (lower-rank noise covariance Σ).

* They propose an easy idea to introduce more symmetries: 
change weight W to v × W, where v is a new scalar parameter (a single learnable number)
The network output stays exactly the same if you scale W up and v down by the same facto -> continuous symmetry: you can rescale v and W in opposite ways without changing the function the network computes.
Furthermore, if v = 0, the entire layer output becomes zero
During training, the optimizer can break this symmetry by moving v away from 0 (making the layer more expressive) or restore it by pushing v back toward 0 (making the layer less expressive / more redundant).

-> This trick is described as a special case of the DCS algorithm (Differentiable Constraint by Symmetry) proposed by Liu Ziyin in an earlier paper. DCS is a general method that uses artificial symmetries to enforce constraints (e.g., sparsity or low-rank) in a soft, differentiable way.

## Idea
* Do we really need to go through symmetry breaking to find good symmetries? Can we also just restrict symmetries from the getgo and make the search between two symmetries easier?
* Is it that symmetries encourage low rank/compressible solutions and thus its useful for generalization or is it the symmetries themselves?
* Do Python programs have a lot a symmetries and as such provide a good restriction of the hypothesis space?
* Can we search for smallest subnetwork that still behaves like θ? 
* Can we introduce arbitray symmetries in a neural network to kind of find the largest set of symmetries that still let the model get 0 train loss? that should be the most compressible and most generalizing solution...
* For ARC-AGI, check if there are tasks where we can generate a generalizing solution and yet SGD finds a solution that is more compressible but yet more generalizable. That would mean that the search for the simplest network that fits the data is not suffient to finding the most generalizable...


* General principle: EXPAND to find differences, clear seperations in the data, then COMPRESS by deleting unneccessary information? In layers of nn but maybe also in GD...

* Like the selfish gene thinks of genes instead of individums maybe we need to think about neurons. Make each neuron want to surivie, ie not have the relu die. survival of the fittest. make this work...

* So basically LLMs think in termns of direction of the activations. Like human understandable would be to have neurons as features, but (maybe its the same) they give each distinct feature a different direction (and maybe also encode in norm). This is also the case of the superposition stuff of Antrophic. What can we do with this information? Rethink how neural networks work... Each layer representing a sum of vectors? then layer above listens for specific vectors that are present... Maaaybe the brittleness of nn comes from the fact that this cannot be encoded definitely with a basis, but they wing it, expecting the features to be far apart such that no collapse occurs. That is why we can always find adverserial examples... And that might be why neural networks get better as they get bigger, as those kind of hash conflicts occur less often... Can we create models that enforce like real features? Maybe for small datasets those would generalize and be not brittle to adverserial attacks.
-> maybe we can set the amount of directions, and layer does then not output any kind of direction, but a sum of those pseudo basis vectors

# Symmetry Induces Structure and Constraint of Learning
## Overview
Instead of using traditional hard or non-differentiable constraints (like L1 regularization for sparsity, or nuclear norm for low-rank), we can introduce artificial symmetries into the parameterization of the model.
These symmetries then softly encourage the desired structured constraint (e.g., sparsity, low-rank, group sparsity, etc.) in a fully differentiable way, simply by training with standard SGD + weight decay (or noise).
In short: Symmetry = Constraint.


* So when we have symmetries, noise/wd will choose the simplest version that satisfies the symmetry thats the point kinda


## Method
* Every mirror symmetry (reflection symmetry) in the loss function forces a structured constraint on the parameters.
* When such a symmetry is present, and gradient noise or wd is strong enough, SGD has a strong tendecy to solutions satisfying Oᵀ θ = 0 (parameters lie in the subspace orthogonal to the symmetry)

* To implement, the most straight-forward method is to multiply a weight matrix with a scalar v

* More advanced, other matrix multiplcations can be done, such as the hadamard product w_i = u_i × v_i, 


* The key property is that the reparameterization is faithful — the model can still represent the exact same functions as before (no loss in expressivity when the symmetry is fully broken), but the training dynamics now have a strong bias toward the structured solution you want.

## Rescaling Symmetry → Sparsity
* If the loss is unchanged when you rescale a parameter in one direction while rescaling another in the opposite direction, the optimizer (especially with weight decay) prefers to set one of them to zero.

* w_i = u_i × v_i  (element-wise multiplication, Hadamard product)

* there are infinite ways to construct u and v to get the same w. out of all of this solutions, if one is 0, that is a stable point where its hard to move away from

* I dont get it completly (but its simlar to difference of L1 and L2)

* Their hardaman is most similar to L1, but under some circumstances creates even more sparsity.

## B. Rotation Symmetry → Low-Rankness

* Rotation symmetry means you can rotate a subspace of parameters without changing the loss.

* Construct W ∈ ℝ^{m × n}.such that You can apply a rotation matrix R to one set of parameters and R^{-T} (or similar) to another set, and the output remains identical ((e.g., W = A B^T with additional rotational freedom, or more cleverly using coupled rotations similar to double rotation symmetry in attention)

* The optimizer + noise/weight decay favors solutions where W only uses a small number of directions (due to math from paper, ie finding the easiest one in the symmetry manifold) → rank(W) becomes much smaller than min(m, n).

## C. Permutation Symmetry → Homogeneous Ensembling
* Permutation symmetry is by far the most common and natural symmetry in modern neural networks

* In a fully-connected layer with n neurons, you can arbitrarily permute (swap) the order of the neurons (i.e., swap corresponding rows of the weight matrix to the next layer and columns of the weight matrix from the previous layer). The network function remains exactly the same.

* This extends to attention heads, residual blocks, or any set of "identical" components.


A key insight from the paper:
"This theorem implies that a permutation symmetry can be seen as a generalized form of ensembling smaller submodels."
In other words, instead of training k completely different models and averaging their predictions (classical ensembling), the network implicitly trains k identical copies of smaller sub-networks and averages them inside the larger model. This is more efficient and emerges automatically.

ou can think of it as the network discovering:

“I only need 8 truly different feature detectors for this task.”
“I will learn each detector once, then make 10–20 almost-identical copies of it.”

* to break permutation symmetrie, use syre 

* wd pulls towards similar weights, because, I think I explained that elsewhere already.

## Ideas
So general idea is that with symmetries, there are many ways to paramterize a function. Stochasticity and wd favor the simplest solutions then, of the ones that can be chosen to satisfy the trainset. 
-> so the more symmetries are induced, the more wd can choose easier functions... introduce the maximum amount or the right symmetries for arc? I mean the data augmentation part that helps a lot is doing a very similar thing I suppose.


# Remove Symmetries to Control Model Expressivity and Improve Optimization
They kinda argue the opposite of the paper paramater symmetry potentially unifies deep learning theory, in that they pose that symmetries are low rank solutions that make the model less rich (I suppose thats just wrong with what else I learned)

* Once in a neural network two neurons compute the same feature, backprop can not differentiate them anymore (as the gradient is the same)-> the gradients stay equal and thus the model behaves like a smaller model. (does not go well with the symmetry breaking behavior we indentifed in the previous paper)

* Neural networks are overparamteriazed, many different weights can give the same solution. Among all those that do get a good solution, wd selects. It selects by lower norm. Lower norm happens to align with more symmetry because suppose a1​w1​+a2​w2​=S then for w1​=S, w2​=0 wd cost is S^2, for w1​=w2​=S/2 wd cost is (S/2)^2+(S/2)^2=S^2/2. (L2 penalizes large values quadratically so spreading mass evenly reduces total penalty)

* Once inside a symmetric solution GD can not escape it anymore, because the gradients for the two weights are the same

* Syre fixes this by adding a randomly sampled bias to each weight, that is not changed in training. Thus when wd pushes the weights to symmetry but with the added bias the gradients are still different


# Saddle-to-Saddle Dynamics in Deep Linear Networks: Small Initialization Training, Symmetry, and Sparsity

They study Deep Linear Networks. They say that if initial variance is low, gradient descent goes from saddle point to saddle point of increasing rank. (if variance too high, it is already at global minimun and does not rly learn or smth)

Starts near θ₀ = 0 (rank-0 saddle).
Escapes along a specific "fast escape path" to a rank-1 saddle (all weight matrices effectively have rank 1, corresponding to learning the dominant singular component/direction).
Then to rank-2 saddle, etc., up to a low-rank global min or infinity.

In between the saddle points, they prove that GD takes the optimal/fastest path

They low rank solutions that are already found are persisted, with every saddle point a new independent component (singular direction) is added.

## Greedy algorithm
The paper shows that the saddle-to-saddle dynamics (in the limit of vanishingly small initialization) approximates a greedy low-rank search algorithm.

Intuition: Instead of optimizing the full high-rank matrix at once, the training process behaves as if it is repeatedly solving:
"Given the current residual error, what is the single best rank-1 update I can add to reduce the loss as much as possible?"

The fastest direction corresponds to the largest singualar value.

The paper assumes (and supports with probability statements) that the flow follows the optimal fast escape path with high probability.

Key features:Plateaus: Long periods where loss barely decreases and effective rank stays constant (the network "lingers" near the saddle, because loss is 0 in many directions due to degeneracy of Hessian).
Sharp transitions: Relatively quicker phases where rank jumps and loss drops.







## Idea
Can we make that explicit in the sense that we first have a rank 1 completly shared network where we learn all there is, then we add another matrix etc etc, and freeze everything that is already learned

I wonder, if those papers prove that DNN have a low rank bias, why not such a big bias that even for small amount of data the correct solution is found?

* can we track the rank of the neural net during training to see how it behaves


# Stochastic Collapse: How Gradient Noise Attracts SGD Dynamics Towards Simpler Subnetworks
They reveal a strong implicit bias of stochastic gradient descent (SGD)
that drives overly expressive networks to much simpler subnetworks, thereby
dramatically reducing the number of independent parameters, and improving generalization

They identify invariant sets, or subsets of parameter space that remain unmodified by SGD

They propose that the simplicity bias of neural networks is due to randomness/noise of SGD (contrary to GD?)
-> they identify a novel perspective on the source of SGD’s implicit bias

* they introduce invariant sets as subsets of parameter space that, once entered, trap SGD (characterize two such sets that correspond to simpler subnetworks and appear extensively in modern architectures: one for vanishing neurons and the other for identical neurons)
* Reveal a sufficient condition for stochastic attractivity — a process attracting SGD
dynamics towards invariant sets. That is s a competition between the loss landscape’s
curvature around an invariant set and the noise introduced by stochastic gradients
* Through their frameworks they show the importance of a large learning rate during early training


(summary: SGD is biased to simpler subnetworks and those subnetworks can help generalization)

## Strong vs weak directions
A direction is strong if:

Moving along it significantly reduces loss
The data has a clear signal there
Gradients consistently point that way

A direction is weak if:

It only slightly improves loss
The signal is small or noisy
Gradients are small and inconsistent
Intuition
Strong direction = “there’s clearly something to learn here”
Weak direction = “maybe there’s something here… or just noise”

-> Now bring in SGD noise (which gets larger with higher learning rate).

Training update ≈
signal (gradient) + noise (randomness from minibatches)

In strong directions:
signal >> noise → learning is stable → survives
In weak directions:
signal ≈ or < noise → updates are dominated by randomness

-> For those weak directions then, as SGD tends to simpler subnetworks, those neurons that are responsible for the weak directions get 0ed out or similar to other neurons

-> With high lr, we have more noise, so more weak directions fail to survive

-> Weak directions often correspond to small singualr values, fine details or noise in the dataset. If you skip them, generalization is better.

## Ideas
Maybe training of nn ist actually many stages, but with GD they get mixed together. Like idk first its right loss identification then,...
Like first we find the rough right solution and then continously refine, and dont refine too early (premature optimization is bad kinda way)

Maybe optimal structure of a nn is fractal.

* In related work they list all the work that went into indentifing how SGD with its elements works to find flat minima? but we want sharp minima?


# (No paper) Memorization Capacity
Interested in how many neuron it takes to learn a specific trainset. Some results:

 O(n) total parameters) to interpolate n generic points (https://chinmayhegde.github.io/fodl/representation01/)

 Tighter constructions show that m ≈ 4 ⌈n / d⌉ neurons can suffice for many datasets in general position.

## Idea
Train a meta Model on many tasks that takes normal nn and then converts it to one that is generalizing


Meta train a model such that it is unable to learn random data (so we optimize init basically) but is able to learn the task at hand. 
-> interesting if weights can be set in such a adverserial manner that sgd can not converge...?

# Sufficient is better than optimal for training neural networks

The authors argue that, for neural networks, chasing the absolute lowest training loss is often the wrong goal. Their point is that a model can become very good at fitting the training set by also learning its noise and quirks, which hurts performance on new data. They propose a different training style called simmering, which deliberately samples near-optimal weights instead of trying to find one single “best” optimum. In their view, “good enough” can generalize better than “perfect.”

## Simmering
nstead of minimizing loss and stopping, simmering keeps the model moving around near-good solutions, sampling many of them and averaging their behavior.

So they first go with normal Adam to good solution, then they explore the area around and create a ensemble with that


# equivariant networks

* Equivariant neural networks are a class of models designed to handle data symmetries (such as rotations, translations, or reflections) by ensuring that if the input changes, the output changes in an equivalent, predictable way

* Definition of Equivariance: for transformation g: f(g * x) = g * f(x)

* Unlike standard networks that must learn to recognize an object in every possible orientation, equivariant networks understand that a rotated object is the same object, significantly reducing the amount of training data required.

* Example: CNN is translation equivariant


# https://www.sethmorton.com/blog/the_geometry_of_surprise
the author is saying, “not all surprise is the same.” If an AI only records how big the mistake was, it may miss what kind of mistake it was, which makes it worse at learning over time. The proposed alternative is something like a “settling substrate” that keeps more of the geometry of the error signal intact, so the system can learn in a more nuanced way.

Old:
prediction = model(x)
actual     = y

error = abs(prediction - actual)   # just one scalar

if error > threshold:
    curiosity_memory.append(error) # loses detail

New:
prediction = model(x)
actual     = y

residual = actual - prediction     # keep full vector / structure

 store details, not just one number
memory.append({
    "residual": residual,
    "context": x.context,
    "features": x.features
})

later, learn from patterns in the residual
if is_unusual(residual, context=x.context):
    update_specialized_submodule(residual, x.context)



# The Tunnel Effect: Building Data Representations in Deep Neural Networks
Trained neural networks split into extractor and then tunnel, which compresses/plateaus

So similar finding to before. Interestingly, they observe that removing the tunnel increases OOD performance and its better for continual learning.

# Understanding How Nonlinear Networks Create Linearly Separable Features for Low-Dimensional Data

They suppose data that lie on a Union of Subspaces (UoS). That is, data from each class lives approximately on its own low-dimensional linear subspace of dimension r inside a high-dimensional ambient space of dimension d (with r << d).


They have a random layer with quadratic activation function and then a linear layer on top. They show and prove that this setup is enough (with not too large witdh) that the random non-linear layer seperates the classes in such a way that the linear layer can get almost perfect accuracy.

* With quadratic activation function they prove, with ReLu they show empirically

* I didnt get too much into the proof, but basically, suppose class 1 are vectors in the x (a,0) axis, class 2 are vecotors on the y axis(0,b). Then there is no linear seperation between them. If we squre them, we get (a², 0) and (0, b²), so points lie only on the positive side of the axis, but still not linearly seperatable. But if we before that do a random projection, then maybe for class 1 we get (a², 0.01a², a², 0.04a²) and for class 2 (0.01b², b², 0.04b², b²). SO class 1 has large values in positions 1 and 3 and class 2 has large values in positions 2 and 4. then a linear clasifier like v = (+1, -1, +1, -1) can seperate them-> For cars: v · f(x) is roughly positive, For dogs: v · f(x) is roughly negative. The more random directions the more likey such a seperation exists.

* For their empirical results, they use a MCR representation of Cifar. It does this by maximizing the difference between (a) the coding rate of all features together and (b) the sum of coding rates of each class separately. In information-theoretic terms, it encourages compression within classes while keeping classes well-separated in the feature space.

# The Platonic Representation Hypothesis

he authors argue that neural network representations are converging — not just within the same domain (e.g., different vision models becoming more similar), but even across modalities like vision and language.
As models get larger, more capable, and trained on more diverse data, they start to measure distances/similarities between data points (images, text, concepts) in increasingly similar ways. The paper hypothesizes that this convergence is heading toward a shared statistical model of reality

## Key findings:

Larger, better vision models have more aligned representations with each other.
Stronger language models have representations that align better with vision models (on paired image-text data like Wikipedia Image-Text dataset).

Alignment tends to increase with scale and performance. Higher alignment also correlates with better downstream capabilities

## Critisism
* Alignment is still weak — Even the best cross-modal scores were modest (e.g., mutual k-NN overlap around 0.16)
* Metrics are confounded by scale — A 2026 paper ("Revisiting the Platonic Representation Hypothesis: An Aristotelian View") shows that many similarity metrics (especially global ones like CKA) are inflated simply by making models wider or deeper. After proper calibration (permutation-based null models), the strong global convergence largely disappears, though local neighborhood similarity (which points are close to each other) remains significant across modalities. They propose a milder "Aristotelian" version focused on shared local structure rather than a full platonic ideal.
Global alignment (e.g., CKA or spectral measures): After calibration for model size, the overall similarity of the full embedding spaces looks much weaker than people originally thought. The “big picture” geometry (exact angles and distances between all points) does not match very well.
Local neighborhood structure (measured by mutual k-NN overlap):
For the image/concept “cat”:
In Model X, the 10 closest neighbors might be: dog, lion, tiger, kitten, rabbit, etc.
In Model Y, the 10 closest neighbors are almost the same set: dog, lion, tiger, kitten, rabbit, etc. (high overlap).
## More support
* An Information-Geometric View of the Platonic Hypothesis: 
From a Bayesian perspective: When you train a neural network, you're (implicitly) performing Bayesian inference
As the amount of data grows and the model capacity increases (more parameters, more expressive architectures), the posterior concentrates sharply around the true underlying function
Result: Their internal similarity structures (kernels) align because they're all converging to the same optimal approximation of reality.
For sufficiently large/expressive models, this convergence is inevitable.

They also prove a "disunion theorem": If two models have meaningfully different approximation capabilities (e.g., one is much narrower or has a strong inductive bias mismatch), their representations will diverge, and the separation can grow exponentially with more data. This explains why we see strong alignment in general-purpose foundation models but not necessarily in narrow/specialized ones.

* Harnessing the Universal Geometry of Embeddings (Strong platonic representation hypothesis)
Not only do sufficiently large models converge to similar representations of reality, but their embedding spaces are so geometrically similar that you can learn a translation function between them without any paired examples
They introduce an unsupervised method to translate embeddings from one model (e.g., embeddings from Llama-3) into another model's space (e.g., Gemma or even a very different architecture), or into a "universal" latent space.

The method works by exploiting the assumption that the underlying geometry (relative distances/angles between concepts) is approximately universal.


# Information-Theoretic Progress Measures reveal Grokking is an Emergent Phase Transition
* proposes information theory as a taskindependent tool to identify emergent sub-networks in neural networks for mechanistic interpretability

Important concepts:
* Synergy refers to the cooperative behavior between variables as a whole,
where their combined statistical interactions exceed the sum
of their contributions in isolation. 
* Redundancy is the shared information between variables

In this paper, we hypothesize that grokking is a phase
transition caused by the emergence of a generalizing subnetwork due to the collective interactions between neurons
as a whole, which cannot be quantified using pairwise metrics

To
understand grokking, we utilize the O-Information - a multivariate information theory measure that scales to multiple
variables - to quantify the synergy and redundancy in a
network 



# Sparsely Supervised Diffusion (SSD)
Training objective (simplified): Optimize the regression loss (denoising or velocity prediction) only over unmasked positions, where m is a binary mask with masking ratio η (probability of masking).

Idea:
Especially with little data, diffusion models abuse local regularities, and thus often dont have global coherence. With this random masking, they are forced to predict a pixel from far away pixels and thus they also need to model global interaction.

## Ideas
This kind of reminds me of the podcast from dwarkesh with the biologist that said that the brain predicts everything from everything

also reminds me of my idea to mask in the attention masks of transformer


# https://x.com/LodestoneRock/status/2050229072488431706
Overfit to one sample to do architecure research

# https://flappingairplanes.com/
Main goal is to use unconventional methods to increase sample efficiency at least 1000x

# https://www.goodfire.ai/research/the-world-inside-neural-networks#
inner world of neural networks is full of structure that reflects the structure of the outer world
Ie days of the week, and months of the year[3] are circular loops in the activation of LLMs

Understanding this geometry allwos for deeper understanding and possible steering of the models.



## Manifold
A 1D manifold (or 1D curve) in this context is a smooth, continuous path or "string" through the high-dimensional activation space of the neural network. So its basically a function that maps from 1-D to n-D. So you can describe where you are on the manifold only with one number (distance from origin) and this brings you to a point in n-D

In reality the curve is approximated with a spline, isomap, principal curves...

This is why steering along it feels natural: you're moving the model's internal state along the path it actually learned, rather than taking weird shortcuts through empty space.

## Approach
Train a model (image-action model on mountain climber, predicting images from image+action), gather activations for many images and then visualize them by dimentsionality reduction.

Then from those reduced points they fit a manifold. This enables them to move smoothly along the manifold. Which they show corresponds to the car moving up the hill.

In contrast they show that linear probes (that move between start and end state only in a linear way) come along undefined points in between that correspond to blurred images.

Also: In days of the week when they move along the found manifold, the output also change accoringly (ie friday -> saturday -> sunday output prob). In linear probes, if you go from friday to sunday you never get the output of saturday.

Also, when they ignore the internals and only train to find a curve that does well in behavior space, they get the same path.  (https://www.goodfire.ai/research/manifold-steering#)

## Critiques
The shape of the manifold is dependent on which dim reduction method you use and thus maybe arbitrary?

But I think in the paper the authors note that its a 'natural' representation.

## Critique from https://deepmanifold.substack.com/p/single-token-geometry-a-critique

(also links some interesting papers)
Deep Manifolds is his idea:In Deep Manifold, neural networks are understood as learned, stacked, boundary-conditioned manifolds. Activations, outputs, and behaviors are not isolated objects; they are coupled traces of the same learned numerical computation.

Argues that the usage of manifold in goodfire is under-defined and advocates for his more detailed Deep Manifold theory.

Key critique 1: Architectural boundraries are missing. ("A token is not floating freely inside a neural network architecture. It is carried by the architecture and constrained by context, prompt, attention...")

Matters because steering must be done in the conext of the boundaries (ie by modifying the attn?) and not just freely in space. (Not sure how exaclty this can be done, but in spirit its just respecting the architecutre more and not moving freely in embedding space)

Critique 2: He says Goodfire only observe isometry between the geometry in activation and output space. (so that the strucutre in activation space and in output space align). He argues more rigorous theory is needed, citing constrained fixed-point computation of his theory.

Critique 3: didnt understand


"In Deep Manifold terms, the wiser path is not to force the manifold from outside, but to improve the boundary conditions under which the manifold learns itself." (so critique is that their method is too artifical and thus prone to overengineering)

## Ideas
Two questions there.
First how to differentiate nn representations and thus manifolds that generalize vs those that memorize

And then given we have a memorizing manifold can we still get new ideas/outputs from interpolation
And if we have a generalizing manifold, can we even get new ideas??

And the even further. lets say we do their experiment but we train on text. for instance on python files that are nn architectures. Can we then interpolate between them and thus get new architectures?

# https://arxiv.org/pdf/2409.17592
Seems interesting maybe read

# https://www.goodfire.ai/research/vpd-explainer#
Interpretability not just for activations but for weights/architecuture itself.

* They split the weights into simple, understandbale components.

Identify algorithms implemented in attention layers, even when they are distributed across attention heads[2];
Edit the original model's behavior with no training — performing "brain surgery" directly on the model's "neural code";
Recover small subnetworks responsible for specific, abstract behaviors.

## Method
* Decompose a matrix into rank-1 
* One subcomponent should be removeable for tasks where its not needed
* They train another network, the causal importance network. On any given prompt, the causal importance network predicts the minimum number of subcomponents that are causally important to reproduce the model's behavior on that prompt.
* They also adverserially search for subcomponets that break the behavior of the simpler model they identified (stress-tests both the subcomponents and the selection made by the causal importance model)

(There are also relatively few subcomponents: Decomposing all 24 matrices in the network, we identify only ~10,000 subcomponents. In Llama 3.1 8B)

## Pseudocode
# VPD Training Overview (simplified)

# 1. Setup
for each weight matrix W_l in model:
    initialize many rank-1 candidates: U[l, c], V[l, c]   # c = component index
    Delta[l] = small residual matrix

causal_g_net = small neural net  # predicts importance g per (input, component)

# 2. Main training loop
for batch in data_loader:          # batch of sequences
    # Full forward pass (ground truth)
    orig_logits = model(batch)     # using full W = sum(u v^T) + Delta

    # Get importances from auxiliary net
    g = causal_g_net(batch)        # shape ~ [batch, seq, layers, num_components]
                                   # g ≈ 0..1 per subcomponent

    # --- Stochastic reconstruction (easy cases) ---
    mask_stoch = sample_masks(g)   # e.g., keep with prob >= g, or uniform in [g,1]
    masked_W_stoch = reconstruct_weights(U, V, Delta, mask_stoch)
    recon_stoch = model(batch, weights=masked_W_stoch)
    loss_recon_stoch = KL(orig_logits, recon_stoch)   # or CE, etc.

    # --- Adversarial reconstruction (hard cases - key innovation) ---
    mask_adv = adversarial_find_bad_mask(   # gradient ascent on error
        U, V, Delta, g, batch,
        steps=10-30,                       # optimize mask within [0,1] bounds from g
        objective="maximize recon error"
    )
    masked_W_adv = reconstruct_weights(U, V, Delta, mask_adv)
    recon_adv = model(batch, weights=masked_W_adv)
    loss_recon_adv = KL(orig_logits, recon_adv)

    # --- Regularization losses ---
    loss_sparsity = mean( g ** p )         # p>1 encourages few high-g components
    loss_freq     = penalize_overuse(g)    # discourage components used too often
    loss_delta    = small_delta_loss(Delta)
    loss_simple   = encourage_rank1_or_low_rank(U, V)

    total_loss = w_adv * loss_recon_adv + w_stoch * loss_recon_stoch \
                 + reg * (loss_sparsity + loss_freq + loss_delta + ...)

    optimizer.step(total_loss)  # update U/V/Delta + causal_g_net

(note the adverserial loss here: Within respecting the choice of g, they search for combinations of subnetworks that break the reconstruction. Enusures that the high-g components are actually sufficient, and low-g ones are truly save to remove)

## Results
Targeted edits, ie changeing how emojis are written
Track a prediction across layers, identifing relevant subcomponents.

## Ideas
One thing I find interesting here is the adverserial loss. The most general idea here is that you have n choices, you constrain those choices with a model and then inside the contrainst, adverserially search for valid combinations that are still bad.
-> how to use this for arc? let a model only predict top3 and then inside them adversially search or smth? mhm 


# Symmetry in language statistics shapes the geometry of model representations

Main hypothesis: representational manifolds have a universal origin: symmetry in the statistics of natural data. (Note, like the other paper...)

## Preliminaries
They look at word2vec. Word2vec learns a vector for each word based on co-occurances of words.
It has been show that they " learn to represent the top eigenmodes of a
normalized co-occurrence matrix, approximately equal to
the pointwise mutual information (PMI) matrix:"

PMI(or the approx of it M*) is just the ratio beetween the co-occurance probability Pij and the probability of both words occuring independetly Pi*Pj. (Positive: they occur often together, negative they "repell" so dont like to be together)

Then they say word2vec does PCA on M* = Φ Λ⋆ Φ⊤

The learned embeddings are basically the top eigenvectors (scaled by the square root of the (absolute) eigenvalues)

Okay so now we could build the PMI and then run PCA. But that is expensive. So instead they want to study and predict embeddings without any training.

For that, they leverage special structure in M* (which enables easier analysis than the general PCA).
(One previous structure is Kornecker deltas, which enables Linear analogies like king - man + woman = queen)
-> going more into detail: this previous work wants to explain those linear analogies. They see each word as defined by a binary vector of attributes (is royal, is male, etc). If two words share many attributes, the tend to appear together. This produces the Kornecker structure and thus king - man + woman effectively flips the attributes. (also analogies still work if you remove all direct co-occurances because they come from the global attribue patters ?)

So the authors in this paper pull a similar trick. But instead of concrete attributes, they look at continous underlying concepts, like months in a year, or coutries by longitude.

Based on how language is structures they assume translational symmetry. So for the month example, the difference between Jan and Feb should be the same as Jul and August.
-> If that is true, M* will have a special repeating structure, namely Toeplitz.
-> When you take the eigenvectors of such a symmetric matrix, they turn out to be sinusoidal waves (Fourier modes) — exactly like sine and cosine waves. 
More concretly:
If we assume such a cyclical structure in M*, then when we do PCA on the centered M* the first two PC don't loose any information. And if we project each word to the 2-dim PC, they form a circle (PC 1 is sin, pc2 cos, together in 2d that is apparently a circle. )


## Related Work
"LLMs learn low-order token correlations—
e.g., the pairwise statistics captured by word embedding
models—before higher-order ones. This suggests that large
language models learn contextualized representations and
computational circuits atop coarse low-order statistics of
natural language. "

## 3. Co-occurrence model with symmetry
Goal: describe the representational geometry of words and phrases that share an underlying continuous concept such as the time of year, historical dates, purely by math, assuming a natural symmetry in language.

1. Assumption:How often two words appear together depends only on how far apart they are on the hidden concept (Again, ie 2021 and 2020 occur as often as 1440 and 1441)
2. Given that, words are spaced on a circle (because of fourrier stuff)
they also make other geometric predictions, with slightly different assumptions
Adding more PCs adds ripples

they also have a plot of US states. Here they with theory predcit i.e. a west-east difference and the observations also support that(better for smaller models, here not much else stuff can be learned). Interesting, because even though there is a lot more information that relates states to each other (sport, culture,...), the translational symmetric one seems to dominate. 
-> core thesis of the paper: Even though natural language contains tons of other information (syntax, topics, sentiment, analogies, culture, facts, etc.), the translation symmetry in simple pairwise co-occurrence statistics is strong enough that it dominates and predicts the most prominent geometric structures we observe in word embeddings and even in LLM representations.

(When people visualize embeddings, they usually look at the top 2–4 principal components. These top directions are precisely the ones most strongly controlled by the translation symmetry. The other semantics are there, but they live in higher dimensions and are less visible in the common visualizations.)

## 4. Collective effects control the embedding of space and time
"If the geometry comes from co-occurrence statistics, what happens if we remove all direct co-occurrences between the months? Does the circle disappear?"
-> suprising answer: No, circle still appears.

The key insight is that many other words in the vocabulary are also controlled by the same underlying continuous variable. 
Examples of seasonal helper words:

Winter: ski, snow, Christmas, January, cold, sweater…
Summer: beach, vacation, July, hot, hurricane, ice cream…

These helper words act as bridges. Even if you remove direct “January–February” co-occurrences, both months still co-occur with many shared seasonal words, so the model can still infer their relationship indirectly.

## Discussion
a single unifying principle: pairs of words that correspond to similar time or space co-occur more frequently in text.

-> pretty geometries are mostly a low order statistical property of natural language

In neuroscience grid cells(predict movement), similar patterns have been observed. They think this is due to the translational quality of movement.

## Ideas
"LLMs first learn low-order statistics before higher-order/contextual ones". Can we unlearn the low-order, the spurious cues? And thus only keep higher order ones?
-> with the quote above, this seems kinda like the models first use the simplest low-order correltations to do the task. if that does not suffice (ie if the data is too large), the higher order correlations are taken.
Wow that is a nice view! from that perspective llms are actually doing "finding simplest solution" only that the simple they use is not the simple we want to have. But they explore in ever more "complex" ways in this perspective. so we would need a way to revert that or skip early correlations... Like maybe even though lower order correlations do help in predicting text, we would not ever want to use them, as everything can be done with higher level order interactions...

those "low/high order" i guess is directly translated to the eigenvectors etc
(I should get more intuition for PCA)

translation symmetry means that only relative distance counts?


In Figure 1 they have plots that compare theory to observation in Gemma. Well, ideally we would have a training method such that make the empirical observations match the theory, such that we get nice and clean representations...


Model LLMs as having a set of concepts (tokens, but also more higher level concepts) that are related with each other. Wolfram alpha style kinda.

So they say that the low-frequency patterns dominate (I think the trans symmetry). But I guess that is not what we want. We would like to have those concepts, but in unrelated topics such as talking about sport, those patterns should not occur anymore.
Maybe we need to disentangle concept from the magnitude of the concept. The magnitude of the concepts seems to be related to the spurious cues -> works in most cases but if you rely on it, on the actual cases that require reasoning it fails. (like the paper that removes magnite from embeddings, but more principled)



# snimu omouamoua on x
SGD as evolutionary search in the loss landscape: 
Models learns on batch N. If what it learned on batch N is (more or less generalizable) it will then survive the gradient on batch N+1
## Ideas
With this framework, would it be then interesting to make two consecutive batches as distinct as possible? Like adverserial batch selection...
Can we train on cifar in such a way that we have a ordering that generlaizes better?
But mhm isnt that just like taking average? like if we combined the two batches...


# EVERYTHING, EVERYWHERE, ALL AT ONCE: IS MECHANISTIC INTERPRETABILITY IDENTIFIABLE?

LLMs implement algorithms. So there is a abstract algorithm (what) and the nn implements it in its neural activations (where). 
So they ask two questions regarding if there are unique MI explanations for what the models compute. Suppose a NN is trained on XOR. Then they have two questions: 

1. (What-then-where) Use candiate algorithms (i.e. NAND OR A AND B (wrong but ye)) and then search in the nn for circuits that match that behavior
2. (Where-then-what) Find circuits, then interpret what algorithm a circuit implements.


## Method
They use small MLPs such that they can enumerate every possible combination of MLP and the task is learning simple bool functions like AND.

They link circuits direclty with algorithmic interpretations. For instance, a binary variable F0 can be assigend to a single neuron, with a threshold at 0.
Or only specific intervals of neuron can be maped, i.e. from 0.1-0.3 it means 1, for -0.1-0 its 0. Parts can have no meaningful interpretation
For variables with more states, you also need more neurons

## Findings
Systematic Non-identifiability:

What-then-Where: For a given algorithmic interpretation there are many circuits that do that

Where-then-what: For a given circuit, there are many possible algorithmic interpretations.



# All Circuits Lead to Rome: Rethinking Functional Anisotropy in Circuit and Sheaf Discovery for LLMs

## Circuits vs Sheafs
Both are sparse subgraphs (DAG) based on the residual stream(additive updates from attention heads and MLPs across layers)

Circuits causally contribute to a prediction, but if you remove all non-circuit parts the prediction is not necessaryl recovered (Note, so sufficient but not ...)

Sheaf is circuit but more, can standalone execute the task

## Core idea
They show that there isn't just *the* circuit or sheaf for a specific task (i.e. indirect object identification). Instead, the paper shows that a single task can be supported by multiple, structurally very different (low-overlap) circuits/sheaves, each of which is faithful (performs the task well), sparse, and complete (works in isolation).

* This is not just "backup heads" that activate only when you ablate the main ones (as in prior work like the IOI paper or the Hydra effect). These alternative mechanisms coexist and independently support the task during normal operation.

# Method
They agument sheaf discovery, by first discovering one good sheaf. Then, in subsequent runs, penalize reusing its edges, forcing the optimizer to find divergent solutions.

## Results

Result on classic benchmarks (especially IOI task in GPT-2 small): Multiple sheaves with near-zero edge overlap (e.g., IoU ~4%) but identical high performance (100% accuracy)

The more sheafs they identify the more distinct the sheafs also get (so the longer you run the discovery, the more diverse sheafs?)

They found an extremely minimal sheaf for the IOI task that only needs 3 edges (connections in the graph) to fully solve the task on its own -> If you remove any one of those 3 edges (forcing the discovery process to avoid it), the method can still find other high-quality sheaves that perform just as well. -> None of the three edges is indispensable across the broader space of solutions. -> This undermines even a weakened version of the "unique mechanism" idea (e.g., "there are a few canonical essential components, plus optional auxiliaries")

In larger models the effect should become more pronounced.

## Explanation
Exponentially many sparse subgraphs. Due to the geometry and redundancy in high-D space, lots of these subsets end up being functionally equivalent (they move the representation in the right direction for the task). This creates a combinatorial explosion of low-overlap (structurally different) but high-performing solutions

* They also mention the subset sum problem -> res stream is additive with many components, and there are exp many ways to sum up to one number if the components can range freely.

## High-dimensional superposition in LLMs

Res stream has limited dim, but Models need to represent far more features. So in the res stream vector they represent features in near-orthogoal dimensions, which are plenty in high-dim space. 
When vectors are not perfectly orthogonal, it means that if you want to increase one feature, that will leak into other features at least a tiny bit. 
-> llms then rely on nonlinearities to filter/clean up those interferences

* Superposition only occurs though for features that co-occur rarely. When features whould occur together often, the interfernce would be too much

* Most SAE features fire only rarely, so they are sparse

* No paper determines exact amount of features they represent (?Evidece: Relu are needed for superposition to occur) but tens of thousands or millions maybe per layer

## Ideas
The Superposition rarley with dense feature: Explains maybe the ruggedness of llms, because even for sparse features, with many tokens the overlap will be a lot and in those cases the prediction will fail at least somewhat I suppose.

* They don't quite answer if in a normal model for a given prediction the different sheafs all combine for a precition or if one sheaf is taken etc etc

# Mechanistic Interpretability with Sparse Autoencoder Neural Operators

They invent an extension for the SAE. Instead of SAE which maps the res vector (m) to one large sparse vector of size , they map to m*p large vecotr. Basically they thus also have p concepts, but with m they also say where the concept is activated. They apply sparstiy both in concept space (normal SAE) and also in position space (such that concepts dont activate everywhere).
## Idea
Not that interesting per se, but the idea behind it seems interesting. Their approach is also just a vector, but they interpret/group it together differently and also they add a another loss function. This seems generally how things work, everything is a vector and the interpretation is what counts.


# Geometric Factual Recall in Transformers

They explore how factual recall works in LLMs and propse that LLMs do it differently than previously assumed.

## Previous explanation (algebraic)
The facts are stored in the MLP. The MLP then basically is a giant lookup table, that stores every fact individually. The embedding would then be the key to ask questions like "Where was Alice born?". MLP then fires when "Alice" and "born in" is activated and gives out the fact.

## Their explanation (geometric)
They see the emebdding as doing most of the work instead. The embedding has many directions, each of which code for a specific attribute, like "occupation" or "birthplace". Then during inference, when a attribute is asked like "born in", the MLP recognizes that and removes all attributes that do not match this question.

## Results
They provide evidence that for N persons with R Relations, d = O(R log N) sufficies. Each person is embedded as a superposition of its R relations. 
O(R log N) because we have R relations

For multihop queries, without CoT, either the embedding must get really large (All hop combinations present) or the MLP must get large ()

For multihop queries with CoT, this does not apply, instead  d = O˜(R + k) suffices for any k-hop query. (: Alice → mother_of → [generates "Charlie"] → born_in → [predicts "Berlin"]) (+k Encoding the current position in the reasoning chain (which hop you are on), Distinguishing the different hops so the gating and selection still work reliably across the full sequence.)

They empirically verify that the threshold emerges with GD (Models succeed around d = Θ(R log N), failing below it) and they do stuff like causal interventions to verify.

Also they freeze a trained MLP swap to entirely new random bijections, and reinitialize subject embeddings to the new superpositions. The model achieves high accuracy without retraining the MLP — proving it learned a general relation-conditioned extractor, not specific fact pairs. (generic selector ("whatever is in block 1, give it to me when relation=born_in").)

Also You can decode "Alice’s occupation" directly from Alice’s embedding with a simple linear probe. The information is already there in superposition.
## Ideas
* General theme I observed now is the following view: An embedding in the res stream stores a number of features, and the MLP with the RELU then distentangles the Superposition and keeps only the relevant features.

* Maybe a good view is that a embedding vector in the res is a superposition vector, ie many directions. But the directions need to be interperted! And that is the MLP, so each MLP interpretes each dir and based on some key it then selects the good directions/features... (A bit washy, need to think about that a bit more)

# Do Sparse Autoencoders Capture Concept Manifolds?

Traditional Linear Representation Hypoethesis (LRH) assumes concepts are represented as independet directions in activation space (clean vectors you can add and subtract).

Recently is became evident that instead cncepts live on low-dim manifolds.

Here they check if SAEs (which assume LRH) can still be useful to reveal the geometry (they are)

## Gemoetric Note on Prior Negative Results for SAEs

1. Across runs, the dictionary of SAEs differ. They say this is because each run makes a different tiling of the underlying geometry
2. Steering features is often brittle. They say this is because the steering is done linearly, often pushing the activation then off the manifold
3. Automated interpretability of SAE features is brittle. They argue that in isolation, the underlying object is missed thus inspecting linear directions is bound to fail

(LRH only directions, geometry also takes distances between points into consideration)
## Three ways SAEs can capture the geometry

1. Shattering/Tiling: Each feature in the SAE corresponds to one point on the Manifold.
(one SAE feature is one point on the manifold, they don't overlap and dont fire together)
2. Compact Capture: The SAE features act as a coordinate system for the manifold
(i.e. 3 features which linear span is the entire true manifold. The 3 features always fire)
3. Dilution: Mixure between the first two, most common in practice.
(Features 1-3 activate somewhat for low temps
Features 2-5 activate for medium
Features 4-8 activate for high
Overlap is messy. Any point on the manifold activates 4–5 features, but different combinations.)

-> Dilution helps explain why SAEs can feel simultaneously illuminating and unsatisfying. A single SAE direction may pick out a meaningful local region of a manifold, but the manifold itself is distributed across many such directions.

## Representation as Additive Mixture of Manifolds
In LRH each concept is a direction scaled by a coefficient. AMM generalizes this such that each concept is a manifold and thus a vector in the res stream is a sum/superposition of manifolds, or more specifically:
$$\mathbf{x} = f_1(m_1) + f_2(m_2) + \dots + f_S(m_S) \quad \text{where } |S| \ll m$$
Each $  f_i(m_i)  $ is a point on a smooth manifold $  M_i  $

## Results
Develops method to search for groups of features that collectively recover the full structure.

Central observation: A SAE cpatures a manifold well, if a small set of features has a linear space to recover all points on the manifold and if the encoder selects this small group every time when a input that is on the manifold is given.

* Difference of intrinsic dimension of the manifold and the dimension of the linear space it lives in: 
* * A straight line (temperature) has intrinsic dimension $  d_i = 1  $, ambient dimension $  k_i = 1  $
* * A circle (days of the week) has intrinsic dimension $  d_i = 1  $ (one angle parameter), but ambient dimension $  k_i = 2  $ (it needs 2D space: $  x = \cos\theta  $, $  y = \sin\theta  $).
* * A helix or Swiss roll has intrinsic dimension $  d_i = 2  $ but often needs $  k_i = 3  $ to embed without self-intersection.

When the number of features in the SAE is around k, that is optimal and compact capture occurs.
When the number of features goes up, SAE is not restricted anymore and it can assign different features to different regions of the manifold, so tiling.

(Tiling is observed in neuroscience)

## Capture Manifolds from SAE (Ising Model inspired)
The group of SAE features associated with a manifold needs to be identified. They use co-occurence, i.e. which features fire together or don't like to fire together.

However, raw co-occurance confounds two (or even 3) sources of statistical dependence: 
* a structural co-activation (atoms that span or tile the same manifold) (i.e. concept Monday and Tuesday for the "Days" Manifold)
* b correlational co-occurrence (concepts that tend to appear together in the data)(i.e. concept Friday and relaxing for the "Days" manifold).
* Universal features: Features that fire almost everywhere (they pollute raw correlations).

They use Ising model inspired to distentangle that.

First they binarize the SAE features.

Then to filter out universal features, that is easy because they are captured by h, so will not be included in J.

Next up they need to differentate a and b, so that they can find the concpets that are part of the manifold. 
Consider the example Days of the Week, with Monday and Friday strongly negatively correlated and "relaxing" strongly correlated with strongly with Friday and strongly negatively with Monday. (fully connected 3 node graph)
-> The Ising model when fitted disentangles this graph. 
-> Basically, the model wants to find the easiest explanation. And as the relationship between "relaxing" and "Monday" can be explained because "Friday" and "relaxing" as well as "Monday" and "Friday" are strongly related, it removes the unneccassry edge from "relaxing" and "Monday"
(Ising model asks, what can we learn about a pairwise relationship, if we have already fixed all other variables? If the fixed variables already explain the relationship well, we don't need a edge)


Thus the have a cleaned weighted feature graph. Next, they use unsupervised clustering (such as the Louvain or Leiden algorithm, or Spectral Clustering) to detect communities. (In network science, a "community" is a cluster of nodes that have a high density of internal edges among themselves but very few edges pointing to the rest of the network.)

-> The "Days" Clique: Because Monday, Tuesday, Wednesday, etc., all have strong structural links to one another (positively or negatively), the algorithm sees them as a tightly-knit, self-contained community. (This is then the "Week" Manifold)

-> The "Relaxing" Node: Since "Relaxing" only has a single link pointing to Friday and zeros everywhere else, the community detection algorithm naturally leaves it out of the "Days" community.


## Ising Model
$p(\mathbf{s}) = \frac{1}{Z} \exp\left( \sum_{i<j} J_{ij} s_i s_j + \sum_i h_i s_i \right)$

-> So this is a prob distrubtion that says "the probability of s (a specifc firing pattern of the SAE) is ..."

* Disentangles interaction between individuals and overall forces

* If J_ij for s_i,s_j is high, this means that if s_i is active, s_j will likely also be active and vice versa. h is the overall streght, like a bias

* It is common in statistical physics to assume such a more complex distribution, fit it to data, but then only take the part that you want from it.

* exp to turn the energy into a probability, lower energy becomes exponentially more probable, standard in the field.

* It explains away the effect of universal features via the h_i fields (high h_i for features that fire a lot, but low J)

* The formula above is the pairwise ising model. there are extension to k-th order like $E = -\frac{1}{6}\sum_{i,j,k} K_{ijk} S_i S_j S_k - \frac{1}{2}\sum_{i,j} J_{ij} S_i S_j - \sum_i h_i S_i$

* Can think of ising as "after I account for/know all other features, is there still a relationship between two features?"

* Ising model is based on local view: "This equation proves that if you want to predict the state of feature $i$, you only care about features where $J_{ij}$ is not zero. In graph theory, these are its immediate neighbors (called its Markov Blanket). If a feature is not an immediate neighbor, it completely vanishes from the equation."

### Hauptsätze der Thermodynamik

0. Hauptsatz: Zwei Systeme, die im Energieaustausch zueinander stehen, immer einen thermodynamischen Gleichgewichtszustand anstreben. Das heißt, dass sich die Zustände der Systeme in Bezug auf Temperatur, Druck und Volumen angleichen.

1. Hauptsatz: Energie kann weder erschaffen noch vernichtet werden. Energie lässt sich nur in verschiedene Formen umwandeln oder übertragen. In einem geschlossenen System ist die Energie deshalb immer konstant.

2. Hauptsatz: Übertragung von Arbeit in Wärme immer möglich, von Wärme in Arbeit allerdings nie zu 100%, dieser Prozess ist irreversibel
Energieübertragung läuft immer nur vom warmen zum kalten Objekt
-> Je höher die Entropie eines Systems ist, desto mehr Anordnungsmöglichkeiten der enthaltenen Teilchen gibt es. Energie fließt dabei immer nur in die Richtung, in der sie die Entropie erhöht. Entropie (das Maß für die Unordnung) in einem abgeschlossenen System bei spontanen Prozessen immer zu

3. Hauptsatz: Ein Stoff kann nicht auf den absoluten Nullpunkt runtergekühlt werden (nur bei perfekten Kristallen mit unendlicher Ausdehnung möglich. Sobald die Gitterstruktur des Kristalls einen Fehler oder einen Bruch aufweisen würde, hätte ein Teilchen mehr Platz als die anderen. Damit wäre auch seine Entropie größer.)


### Ising Model Phase transitions

* Every thermodynamic system tries to minimize: $$F = E - TS$$, Helmholtz Free Energy equation, E is the internal Energy T is Temperature, S Entropy 
Erklärung:
-> Minimierung der Helmholtz Energie ist äquivalent mit der Maximierung der Gesamtentropie des Universums (2. Hauptsatz der Thermodynamik)

By changing the temperature ($T$), we tilt the scales of this tug-of-war. The Ising model reveals three distinct regimes:

1. Low Temperature ($T < T_c$): The Ordered Phase (Ferromagnet): Random thermal shaking (TS) is low, E wins. (If a few elements flip, they get flipped back into order by their neighbours)
2. High Temperature ($T > T_c$): The Disordered Phase (Paramagnet): Spins flip wildly and randomly. Even if a small cluster of spins tries to align, the thermal noise immediately tears them apart. Up and down cancels out
3. The Critical Point ($T = T_c$): The Phase Transition: Here Energy and Entropy are perfectly balanced. It becomes hyper-sensitive, exhibiting spectacular properties: 
    * Long-Range Correlations: A spin flipped on one side of the material can instantly influence a spin on the exact opposite side, spanning macroscopically large distances. (Infinite Correlation Lenght, ripple effect) 
    * Fractal Spin Clusters: If you look at the grid, you will see clusters of aligned spins of every single size scale—from a tiny cluster of 3 atoms to a massive continent of millions of atoms. If you zoom in on a cluster, it looks structurally identical to the whole system (scale invariance).

Tc is determined by the strenght of the interactions between elements (ie J), The number of neighbours, and the spatial dimension d of the network 
    * For 1-D (Chain) Tc=0: (boundries can not be flipped back)
    * For 2D (Grid) $k_B T_c \approx 2.27J$
    * For 3D (Cube) ($k_B T_c \approx 4.51J$)

### Critical Brain Hypothesis
* In neuroscience, evidence suggests that the healthy human brain operates right at a critical state.

* If the brain is too cold ($T < T_c$), it is too rigid—neurons lock into repetitive patterns (like an epileptic seizure), and it cannot process complex information.

* If the brain is too hot ($T > T_c$), it is pure white noise and chaos—neurons fire randomly, and no thoughts can form.

* At $T_c$, the brain achieves maximum information storage, a massive dynamic range to process inputs, and the ability to instantly adapt to new stimuli. Neuronal "avalanches" cascade across the brain at all scales.

* Studies tracking sleep deprivation and anesthesia show that as you get tired, or as drugs take effect, the brain drifts significantly away from the critical state into a sub-critical (overly ordered/sluggish) regime.

### Relationship to Buddhism of Criticality
* Criticality is the literal mathematical realization of the Middle Way. It avoids the dead rigidity of the frozen crystal state (Eternalism/Dogmatism) and the meaningless white noise of the gaseous state (Nihilism/Chaos).

* Interdependence (Pratītyasamutpāda): Buddhism asserts that nothing exists in isolation; all things arise in dependence upon multiple causes and conditions. At $T_c$, the "Infinite Correlation Length" mirrors this perfectly. Separations blur. You can no longer describe the behavior of Atom A without describing the entire network. Every part of the system becomes intimately and causally woven into every other part.

* Impermanence (Anicca) and Emptiness (Śūnyatā)
At the critical point, macroscopic structures (massive clusters of aligned spins) constantly form, dissolve, and reform in a split second. If you point to a cluster of "order" and say "there it is," it has already dissolved into chaos, only to emerge somewhere else.
The structures have no permanent, inherent essence—they are "empty" of independent existence, yet they form the vibrant, ever-shifting reality of the system.



x
## Ideas
* What other representiational ideas are there other thatn Additivatve Mixture of Manifolds and LRH (what other generalizations)

* I think the difference between intrinsic dimension and ambient dimension is quite interesting. What is optimal from a NN sense? like what generalizes best, both low or low intrinsic but high ambient? etc

* k-th order ising model explodes in paramter size and also is similar to taylor expansion imo. can we use that to model llms? like maybe the more paramters a llm has, the higher k-th order ising model it can represent

* -> or can we restrict a model such that it only has orders of idk 4 and 5 but not 2? is that maybe how nn work? can we skip lower level associations such that we only get the higher order ones that are more like reasoning? 

* Can we just treat the res stream as binary numbers, and run the Ising model on that?

* To bypass SAE, use Mapper algorithm, Geodesic Manifold Learning (geodesic distance—the shortest path between points only by traveling along the data graph), Pullback Geometry, Local Intrinsic Dimensionality (LID) Profiling, https://gemini.google.com/app/a9597c1e6fe1949f?hl=de 

### Ising

* Ising method interesting for generalization? Seperated spurious from real?

* There are other methods that generlalize correlation (Partial Correlation, Conditional Mutual Information, PC Algorithm)

* Hopfiled networks, neural populations, protein folding, Restricted Boltzman Machines

* Dynamic ising with time evolution

* Its not a causal method. Why not?

* Ising model assumes conditional-independece assumption, if you doubt that the results might be misleading

* For Gaussian data, a common “fix” is to use the precision matrix / inverse covariance, because zeros there encode conditional independence

* Even with pairwise ising, strong collective effects can emerge

* Energy landscape view of brain states: The Ising energy function defines a “landscape” of possible activity patterns. The brain’s dynamics can be seen as wandering on this landscape — with attractors corresponding to different perceptual, cognitive, or behavioral states.

* Ising defines a energy function, so it gives each possible state a energy
-> Concepts like attractors (basins) or basin jumping can be looked at
-> real brain often has a few deep, wide basins that dominate activity, but system can switch between them (Metastability, realtively stable yet still flexible)

* Muli-layer ising models (still only one timestep), hirachical ising models (not just connection between neurons but also groups of neurons etc)

* Restricted Boltzman Machines introduce hidden neurons that are connected to the observed neurons. This makes it easier for higher order interactions to emerge

## More on Ising

* Adheres to maximum entropy principle (Maximizing entropy ensures that no additional structure is imposed beyond the stated constraints. Any lower-entropy alternative would encode extra regularity not required by those constraints and would therefore amount to introducing unsupported information.)


# Deep Boltzmann Machine

## Energy
Defines a energy function for all possible states -> Scalar. Chosen is such a way that sampling later becomes tracktable.
* The form comes directly from the Ising model and Boltzmann distribution in physics. Each term -x^T W y represents the interaction energy between two groups of binary variables (like spins in a magnet).
It is quadratic (bilinear)
## Sampling
Gibbs sampling then used to do inference, start with a v0 and then do a number of steps. Gibbs sampling is a smart way to not have to calcualte a really complicated calculation
## Training
use real data and made up data and then do a contrastive loss.

# Meditative absorption shifts brain dynamics toward criticality
Researchers tracked highly experienced meditators during a 10-day retreat using EEG to measure complex signal dynamics like Lempel-Ziv complexity, sample entropy, and chaotic Lyapunov exponents. They compared standard mindfulness of breathing to deep, refined states of meditative absorption known in Buddhism as the Jhānas.  The Finding: The paper demonstrated that entering deep Jhāna states causes a massive, volitional shift toward a metastable, near-critical regime. As the meditators slipped into deep absorption, their brains exhibited minimized chaoticity alongside maximum neural signal diversity.

# Meditation Can Reshape Your Brain Activity
This study used high-resolution Magnetoencephalography (MEG) and machine learning to scan the brains of 12 Tibetan Buddhist monks at an Italian monastery.

Vipassana pushed the monks’ brains directly into the sweet spot of brain criticality. By widening the "flashlight beam" of awareness, the brain achieved a state of hyper-flexibility, becoming perfectly poised to process incoming information without getting stuck.  

Samatha, on the other hand, actually distanced the brain from the critical tipping point. By narrowing focus down to a single point, it suppressed chaotic fluctuations, pulling the brain into a deeply stable, sub-critical state of internal quietude.  

# Simulated Annealing Algorithm
Exploration: The algorithm starts at a high temperature, randomly trying new solutions.
Escaping Traps: It accepts worse choices early on to avoid getting stuck in local dead ends.
Exploitation: As the temperature drops, it focuses strictly on refining the best overall solution.

* To guarantee you find the global minimum, the temperature $T$ at step $t$ must be lowered incredibly slowly, following a logarithmic decay (means that number of steps required grows exponentially)

# Chaotic Lyapunov Exponent: The Butterfly Effect Metric

Imagine tracing a single path of brain activity through time. Now imagine a second path that starts almost exactly in the same spot, separated by just a microscopic nudge. The Lyapunov Exponent calculates how fast those two paths diverge from each other.

The Logic: * If $\lambda$ is negative, the two paths will quickly slam back together. The system is highly stable, rigid, and dampens all disturbances.If $\lambda$ is positive, the two paths will violently spiral away from each other exponentially fast. This is the Butterfly Effect—a tiny change at the start leads to a completely different future state.

$\lambda$=0 is the defintion of criticallity

# PC Algorithm
Start with a complete graph between all variable.

1. The Skeleton Phase: Iteratively removes edges between variables if they are found to be independent given a subset of other variables. (i.e. test of given X, Z and Y are still correlated. If not, remove the edge: $Y \perp Z \mid X$.)

2. The Orientation Phase (Assigning Cause and Effect): Analyzes specific shapes within the skeleton (like "V-structures" or colliders) to determine the direction of the edges. Applies a set of logical rules (such as the Meek rules) to orient the remaining edges without creating cyclic paradoxes

## Compared to Ising
Directed vs. Undirected
Binary vs. can be binary but also continous

Discovering the skeleton of a binary network using conditional independence tests is mathematically equivalent to Ising Model Selection (PC then goes a step further)




# Exploratory Causal Analysis (Causal Discovery)
Focuses on extracting the underlying cause-and-effect mechanisms from observational data. Unlike traditional machine learning—which maps correlations—ECA determines why variables interact and can predict the outcomes of unseen interventions.

The ultimate goal of ECA is to map relationships into a Directed Acyclic Graph (DAG), where nodes represent variables and arrows represent the direction of causality (e.g., \(X \to Y\), meaning X causes Y)

## Constraint-Based Algorithms

It begins with a fully connected network and systematically "prunes" or deletes edges if two variables are found to be independent given a set of other variables. (Note: seems quite related to ising)


Common Algorithms: PC Algorithm (Peter-Clark) and FCI (Fast Causal Inference)(extension to PC, includes potential hidden cofounder that is not measured) for cases with unobserved confounders.

## Score-Based Algorithms

These algorithms treat causal discovery as an optimization problem.

They define a mathematical scoring function (e.g., Bayesian Information Criterion) to evaluate how well a hypothetical DAG fits the observed data. The algorithm then performs a heuristic search (greedy search) to find the graph that maximizes the score.

Best for: Scaling causal discovery up to high-dimensional datasets while ensuring the final graph structure is stable and mathematically sound.

## Asymmetry / Functional Causal Model-Based Algorithms

These algorithms look past simple conditional independence by exploiting the asymmetry between cause and effect

By analyzing the distribution of the data and its error (noise), the algorithm determines which direction fits the physical reality (e.g., distinguishing between \(X \to Y\) vs \(Y \to X\)).
(LiNGAM (Linear Non-Gaussian Acyclic Model) and ANM (Additive Noise Model).)

## Hybrid Algorithms
These combine the strengths of both constraint-based and score-based methods

They typically use a constraint-based approach to narrow down the search space (creating a skeleton of the graph) and then apply score-based metrics to find the best possible causal directions.

Common Algorithms: MMHC (Max-Min Hill-Climbing) and SADA.


# Linear Causal Representation Learning by Topological Ordering, Pruning, and Disentanglement
tackles a difficult problem in machine learning called causal representation learning (CRL). The goal is to discover the hidden causal factors that generate observed data.

Develop a CRL method and then they also apply it to an LLM. On its activations. Early research.

# https://towardsdatascience.com/causality-an-introduction-f8a3f6ac4c4a/

Need for causality, not just correlation

A key distinction is:

Correlation is symmetric: if X is correlated with Y, then Y is correlated with X.
Causation is directional: if X causes Y, changing X can change Y, but changing Y does not necessarily change X.

## Structural Causal Models (SCMs)

Pearl's framework represents causality using Structural Causal Models, which consist of:

Directed Acyclic Graphs (DAGs) — diagrams where arrows represent causal influences.
Structural Equation Models (SEMs) — equations describing how variables generate one another.

A Structural Equation Model (SEM) is a collection of equations that represent the causal mechanisms generating a system: X i ​ =f i ​ (causes of X i ​ ,U i ​ )