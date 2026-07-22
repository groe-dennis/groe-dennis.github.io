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

# The Neural Compiler: Program-to-Network Translation for Hybrid Scientific Machine Learning

For physics nn modeling. They have data and want to determine a paramter in a function with that. But they dont just want a nn, that has to learn everything in its weights and is not rly interpretable. Instead they already have a function and just want to learn a speciic part of the function.

To do so they take the function, compile it into a pytorch model and then train that model.

# From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence

They say: Used to train for in-distribution perfomance. Now the goal has shifted to broad generalization to unseen tasks. Thus data is not about the data itself, but what the model can extract from it in order to learn general behavior
-> Data selection
(existing theory here contradicts empirical observation)

### Data processing inequality (DPI)
No computation, statistical operation or algorithm can increase the information content of a signal. So in NN layers the information is either kept (perfectly reversible computation) or partially destroyed. Intermediate representations contain no more information than raw input data.
----
With DPI, synthetic data should add no additional value/information. Also from that framework Alpha Zero is a mystery as it can learn only by RL on the game rules.

* They instead frame everything as the "amount of structural information a *computationally bounded* observer" can extract from a dataset
-> Here existing notions from Shannon and algorithmic information theory are inadequate (and obscure)

-> They indentiy three paradoxes, that can be justified with Shannon/information theory, yet don't conform to empirical phenomena
### Paradox 1: Information cannot be increased by deterministic processes
For both Shannon entropy and Kolmogorov complexity, deterministic transformations cannot meaningfully increase the information content of an object. 
-> At odds with pseudorandom number generatiors to produce randomness, synthetic data to improve models and derivation of new knowledge by reasoning from axioms, emergent phenomena and self-play like AlphaZero

### Paradox 2: Information is independent of facotirization order

A property of both Shannon
entropy and Kolmogorov complexity is that total information content is invariant to factorization: the information from observing first X and then Y is the same as observing Y followed by X
-> But LLMs learn English text better when ordered left-to-right (functions that are easy to predict in one direction and hard in another)
(Note: Idk, wouldn't that be in independent information? Like a text is not just independent words, its the ordering that gives the meaning in the first place)

### Paradox 3 Likelihood modeling is merely distribution matching
They propse that a computationally-bounded observer can uncover more structure than there is in the data generation process
-> In Conways game of life the data generation is simpe, but what emerges are complex things like gliders. While unbouded observers can just simulate the simple rules, bounded observers need to make use of the emergent structures


To capture those paradoxes, they introduce a new information measure called epiplexity (epsitemic perplexity) = the amount of structural information that a computationally bounded observer can extract from the data
= the information in the model that minimizes the description length of data under computational constraints

(Note: For ie game of life I would expect that the generally smallest model would be the rules itself, and epiplexity pockets of reducability model. Actually mhm no the generally smallest model would need to be inf looped then...)

-> observer dependent: the same
object may appear random or structured depending on the computational resources of the observer. For instance, the output of a strong pseudorandom generator appears indistinguishable from true randomness to any polynomial-time observer lacking the secret key (seed)

High epiplexity data should be data that induces generalizing structural features in the model, for instance induction heads. (However, while it epiplexity measures the amount of structural information it does not say that it will be useful for downstream tasks, i.e. tasks where induction heads are not useful)


## Background
In order to define the interesting, structural, and predictive component of information, we must separate it out from random information—that which is fundamentally unpredictable given the computational constraints of the observer

in 1900s, question was: What does it mean for a uniformly sampled inf sequence of bits to be random?
-> Intuitively every sequence should be random, as they are all equally likely
-> However this goes against certain principles, like the law of large numbers, which poses that limN→∞ the avg should be = 0.5 (so i.e. sequence 111111 doesn't fit that)
-> So they first thought to just take those sequences that pass this and other test. However, "other tests" basically exludes every sequence so nothing would be random

-> To solve the paradox, a sequence is random, if it passes all *computable* tests for randomness. (in terms of gambling: A sequence is random if and only if there is no computable betting strategy that can make an infinite amount of money playing against it.)

### Kolmogorov complexity
K(x) = min{ |p|: U(p) = x }
So the Komplexity of a string x, is the length of the shortest string z that defines a turing machine that outputs x.

Can also we defined on Sets S, so a TM that outputs all members in S.

The conditional complexity K(x|y) is the length of the shortest program that outputs x and halts when provided y as input.

### Martin-Löff random
An infinite sequence is Martin–Löf random iff there exists a constant $c$ such that for all $n$, $K(x_{1:n}) \geq n - c$.
(A sequence is random if you cannot compress its prefixes. For any length $n$, the shortest program to generate the first $n$ bits is basically just as long as the bits themselves (minus some minor fixed overhead constant $c$). There are no shortcuts or formulas to compress it.)
(this is incomputable though, because due to Halting Problem Kolmogorov complexity is incomputable)

To extend to finite sequences:
* $c$-random: A finite sequence $x$ of length $n$ is $c$-random if $K(x) > n - c$.
* Randomness Discrepancy ($\delta(x)$): Defined as $\delta(x) = n - K(x)$. It measures how much a sequence can be compressed

Also, randomly sampled strings are with a high probability are Martin-Löff random:
$$P(K(X) \leq n - c) = P(\delta(X) \geq c) < 2^{-c}$$
(If you set a threshold $c = 10$, the probability that a randomly generated 100-bit string can be compressed by 10 bits or more is less than $2^{-10}$ (about $1$ in $1,024$, or roughly $0.1\%$).)

So if we find a sequence that is highly compressible, with high prob we can state that it was not generated randomly.

Computable numbers like pi or e are not algorithmically random even though at first sight they seem like it

### Cryptography
Cryptographically secure pseudorandom number generatior (PRG) are functions that produce sequences that pass all *polynomial time* tests for randomnes.
-> No fast algorithm can tell the difference between the fake randomness and real randomness.

PRG: take small amount of numbers, produce large amount of numbers that can then not be distingushied in polynomial time from random numbers

(or: no polynomial time predictor can predict he next bit of a sequence better than random)

PRGs central point is a One Way Function (OWF) (easy to compute but hard to reverse)

* While cryptographers care about the massive gap between polynomial (fast) and exponential (impossibly slow) times to ensure security, Machine Learning architectures care about different, tighter resource boundaries.
->Quadratic vs. Cubic time: A highly relevant boundary for Transformer self-attention (also CoT)

### Random vs. Structural Information (Sophistication)
Idea: Capture the structural information in a object as opposed to.
To do so, define Sophistication:

$$\text{nsoph}_c(x) = \min_S \{K(S) : K(x \mid S) > \log|S| - c\}$$

where x is element in S

-> The length of the shortest program that describes the structural/non-random part of x (how complex the string is given we remove every random part)

-> the smallest Kolmogorov complexity of a set S such that x is a random element from that set 

very regular string → low sophistication
very random string → low sophistication
“interesting” structured string → high sophistication

So Sophistication is the smallest Ko Complexity of a set S, given some contraints
-> We take a look at many (or all) sets of binary strings. From all those sets, we only take those where the length of the shortest program that outputs x given S is greater than the kardinality of S (minus c), (so its ML random)
-> From all of those, we take the length of the shortest program that produces S

So for the condition it says: Given that the TM already knows S, how complex is it to produce x? And we want ones where its at least somewhat complex.
-> equivalent to say that the TM that outputs x given S (which is a bitstring) is ML-random (the program is equally long as the amount of states it can produce)

It is done by searching for a set/model S such that:

x belongs to S,
S is describable by a short program,
and x is still a typical member of S, meaning it still takes about log∣S∣ bits to pick x out of S.


If x is a special member of S, then it is easy to pick it from S, so K(x∣S) is low, thus it gets filtered from the conditional. 

Basically, we need a TM that outputs stuff like S, and then x needs to be random in S, so if we have a grid with a pattern and like 3 random flipped bits, we can take the pattern in S because then to identify x (so a fixed set of random flips) we still need a lot to identify it.

---
I still don't complety get it tbh. ChatGPT seems to be confused as well. actually in the secodn to last message here https://gemini.google.com/app/15d71bab430a6f7b?hl=de its explained well

---> A string of very high sophistication would be a data string that has very high Kolmogorov complexity ($K(x)$), yet its complexity comes from an incredibly dense, layered set of rules ($S$) rather than from meaningless random noise.

pure radio static: High K, low soph
digits of pi: low K, low soph
playing chess perfectly: high K, high soph

### Why Sophistication is not enough
First, we can never actually find one or prove a specific string has high sophistication.

(You can never prove that any specific string has a Kolmogorov complexity higher than $L$.Because proving a string has high sophistication requires proving it has high Kolmogorov complexity ($K(x) > L - O(1)$), you hit a logical dead end.)

Second, sophistication assumes infintie compute. Thus, many things that look complex to humans might actually have a sophistiication of almost zero. Example fluid dynamics: if a program has unlimited computation time, it doesn't need to describe the whole swirl. It just needs a tiny program containing the basic Navier-Stokes physics equations and the initial starting positions. (Tm can just run those simple steps for a large number of time and reproduce exactly, thus low soph)
### MDL Principle

It answers a simple question: If I have a dataset, how do I pick the absolute best model or neural network to explain it?

The best model is the one that minimizes the total number of bits required to store two things:$$L(x) = \min_{H \in \mathcal{H}} \Big[ \underbrace{L(H)}_{\text{Part 1: Size of Model}} + \underbrace{\left(-\log P(x \mid H)\right)}_{\text{Part 2: Size of Data given Model}} \Big]$$

$L(H)$ — The Cost of the ModelThis is the number of bits it takes to write down the model $H$ itself (e.g., the file size of the Python code or the number of parameters/weights in a neural network).

$-\log P(x \mid H)$ — The Cost of the Leftover Errors
(If a model makes perfect predictions, $P(x \mid H) = 1$, and $-\log(1) = 0$ bits. The data takes up no extra space because the model completely predicted it.)

### Epiplexity
The Dual of MDL (MDL is model selection, Epiplexity is Data selection)
-> You have a fixed, unchangeable computation budget (e.g., you can only afford to train a model for 24 hours). You use this metric to look at a massive ocean of data and select the exact subset of data that fits your budget perfectly.

Epiplexity captures the structural information present to a computationally bounded observer
-> As the computational constraints of this observer change, so too does the division between random and structured content.

Time-bounded probabilistic model ($P$).: A formal defintion of a program that acts as a statistical model under time constraints.
* Evaluation: f you hand the program a piece of data $x$, it must tell you the probability of that data ($P(x)$) and halt within $T(n)$ steps.
* Sampling: If you feed the program a stream of random coin flips, it must output a realistic fake data sample $x$, also halting within $T(n)$ steps.

Now Defintion:
Given a random variable X (a distribution of data, like all text on the internet), they are interested in the best time-bound program that solves:
$$P^\star = \arg\min_{P \in \mathcal{P}_T} \Big\{ \underbrace{|P|}_{\text{Program Size}} + \underbrace{\mathbb{E}\left[\log \frac{1}{P(X)}\right]}_{\text{Average Error Size}} \Big\}$$

From here they define

*Epiplexity*: $S_T(X) = |P^\star|$
-> Size of the program, captures he amount of structural information that a limited observer can successfully extract from the data in $T$ steps.
*Time-bound-entropy*: $H_T(X) = \mathbb{E}[\log 1/P^\star(X)]$
-> expected number of leftover bits needed to compress the data given that model. It represents everything that still looks like random noise to the observer because they don't have enough time to compute the underlying pattern.

Result: with small time budget, epiplexity is low as the model can only see basic structures
As the computational budget increases, what previously looked like uncompressible noise (high entropy) is suddenly revealed to be structured.

pure noise has zero epiplexity and simple patterns have zero epiplexity

$$MDL_T(X) := S_T(X) + H_T(X)$$Think of $MDL_T$ as the total "storage footprint" required by a computationally bounded observer to hold the data—the size of their model plus the size of the errors they couldn't figure out in time $T$.

4) $MDL_{T'}(f^{-1}(X)) \leq MDL_T(X) + |f| + c_2$This is the most critical and interesting property in the list. It is the time-bounded version of a famous rule in information theory: Processing data cannot create new information.
### Pseudorandom number sequences have high random content and little structure
For a fast computer, the epiplexits of a pseudorandom number is very tiny. earlier measures like shannon or kolmogorov do not capture this.

### Existence of Random Variables with High Epiplexity
They prove that they exist, however growing log in data dimension. 
(They say this does not explain the power laws observed in model data scaling, but I actually think this seems intuitive -> from n data you only get log(n) structrual data, kinda like the inverse scaling laws)

Epiplexity typcially grows with the size of the dataset

the epiplexity of a typical dataset is
orders of magnitudes smaller than the random information content

### Conditional epiplexity and time-bounded entropy
In standard machine learning tasks (like training an AI to look at an image $X$ and predict a label $Y$), we do not care about the complexity of generating the image. We only care about the complexity of the relationship between the image and its label.

best fast-running program ($P^\star_{Y \mid X}$) that takes $X$ as an input and outputs the probability of $Y$:

$$P^\star_{Y \mid X} = \arg\min_P \Big\{ |P| + \mathbb{E}_{(X,Y)}\left[-\log P(Y \mid X)\right] \Big\}$$

Conditional Epiplexity ($S_T(Y \mid X)$): The size of the program (or neural network weights) needed to learn the predictive rule mapping $X \to Y$.

Conditional Time-Bounded Entropy ($H_T(Y \mid X)$): The remaining unpredictability of the labels that the model couldn't figure out within the time limit $T$.

## Measuring Epiplexity and Time-Bounded Entropy

How do you measure the "bit length" of a neural network?

Naiive approach: let P be a program that
directly stores the architecture and weights of a neural network and evaluates it on the given data
-> this approach can significantly overestimate the information content in the weights, particularly for large models trained on relatively little data. 
->  Instead, use a more efficient approach that
encodes the training process that produces the weights

-> prequential coding (heuristic but easier to evaluate) and requential coding (more rigorous but harder to evaluate)

Instead of "Turing Machine Steps" they use FLOPs and draw on common training laws (Kaplan, Training a neural network with $N$ parameters on a dataset of $D$ tokens takes approximately $6ND$ FLOPs, evaluating 2ND FLOPs)

## Approximating Model Description Length with Prequential Coding

(Classic approach for compressing the training process of a neural network)
-> basically the area under the curve of the training run above final loss, so 

* Core concept: Synchronized Decoder Game (Sender and Reciver NN)
-> Both initialized the same
-> Sender gets training token Z and calculates the prob
-> Using an arithmetic coder, the prob is compresed into $\log \frac{1}{P_i(Z_i)}$ bits. (the more suprised Sender is, the more bits are required, Shannon)
-> Receiver gets the code and using its own network, perfectly reconstructs Z

Thus, 
$$\text{Total Combined Code Size } L(Z_{:M}, P_M) = \sum_{i=0}^{M-1} \log \frac{1}{P_i(Z_i)}$$
Which is both training data and final model weights

(Good predictions take fewer bits to transfer because like in the game 20 questions, the desired word is already almost identified and fewer bits are needed)

-> Now, this code is the information of the data and the final model weights together. But they desire only the final model weights information

* They define the leftover description length of the data given the final model as its entropy code length under that fully trained network:
$$L(Z_{:M} \mid P_M) = \sum_{i=0}^{M-1} \log \frac{1}{P_M(Z_i)}$$

By subtracting this final baseline cost from the step-by-step training costs, they isolate the model's footprint:$$|P_{\text{preq}}| \approx \sum_{i=0}^{M-1} \left( \underbrace{\log \frac{1}{P_i(Z_i)}}_{\text{Loss at step } i} - \underbrace{\log \frac{1}{P_M(Z_i)}}_{\text{Final Converged Loss}} \right)$$

-> This is the area under the curve of the training run above the final loss, when data is i.i.d

* Pure random noise has low area (so low epiplexity) as model never learns
* Super simple data also has low area (so low epiplexity) as loss instantly drops

This area is measured on the test loss, else the model can cheat by memorization (as is the case in the decoder game, decoder must guess Z before training on it)

### Downsides of prequential
* Information symmetrie (P(x/y) = P(y/x)) only holds in the infinite compute setting and not when time bounded. I believe massive problem as this is the very fact epiplexity depends on
-> prequential only proves that a program of that size exists, not that it can be executed quickly
-> We know it takes exactly $6ND$ FLOPs of time to generate the model weights $P_M$ by running the full training loop token-by-token.
->But if we compress those weights into a compact file of size $|P_{\text{preq}}|$, how long does it take a computer to unpack that compressed file and actually use the model?
->The math of time-bounded Kolmogorov complexity states that unpacking a highly compressed representation can sometimes take an astronomical, exponential amount of time (like trying to guess a password by brute force).
-> (we only get a code for data+model. if we then only use the code part, unpacking to nn weights could take a long time)

* The calculations with the decoder game is a upper bound on Kolmogorov COmplexity, i.e. a smarter algo could compress even more. Then due to math, subtracting the two upper bounds does not give an upper bound for the model

##  Explicitly Coding the Model with Requential Coding

Flaw of prequiential is that it is based on a real dataset. Requential instead ignores the real dataset during the transmision game. Instead it utilizes a Teacher-Student framework to compress the statistical behavior of the model.

* Core hack: the exact identity of the training data points doesn't matter. If you want to train an image classifier to recognize a "cat," you don't need to look at five specific cat pictures; you just need to look at any five realistic cat pictures.Therefore, Requential Coding does not pay bits to compress real data ($Z$). Instead, it compresses a training run that uses purely synthetic, fake data generated on the fly.

* The sender has a sequence of pre-trained "Teacher" checkpoints ($P^t_0, P^t_1, \dots, P^t_{M-1}$) and a student network. Teacher *can* be a model trained on X

* At step i the sender uses the current teacher to sample a synthetic data token. Then similiary as before, this token is compresed given the students predictions. And then normally sent and receiver updates his student

* Due to relative entropy coding (Say you have two Distrbutions, P of the teacher and Q of the studnt and data actually follows P but you want to encode with Q. Then the bits required is defined by the KL divergence between P and Y) 
-> $$|P_{\text{req}}| \approx \sum_{i=0}^{M-1} \text{KL}(P^t_i \parallel P^s_i)$$

-> this can be seen as the area between the curves of student and teacher. In the case that the teacher is static, prequential is an approximation of requential

### Why this solves the mathematical shortcomings
* For prequential we needed to subtract to upper bounds because we wanted to remove the data from the model size. This does not lead to a upper bound. As requential does not rely on data this goes away. The only data sent is the nudging of the teacher to the student, not the data itself

(Prequential Way: The Sender gives the exact street address, house number, and GPS coordinates of a specific house ($Z_i$)
Requential Way: The Sender doesn't care about a specific house. They just want the Receiver to shift their attention toward that neighborhood. So, the Sender just yells: "Go North-East!" The phrase "Go North-East" is incredibly short.)

* Also bcause we directly get a code for the model, we can just use the decoder game to get the nn weights, thus it is time bounded in normal training time. For prequential this only works for the data+model code but that is not what we try to estimate.

(Honestly I dont complety get this)

## How Epiplexity and Time-Bounded Entropy Scale with Compute and Data

Natural assumption: Larger models are more sample efficient.
And: You need to scale both data and model size

Epiplexity grows with compute budget, so it allows to extract more structural information and reduce apparent randomness.

(However there are counterexamples related to emergence.)


##  Paradox 1: Information Cannot be Created by Deterministic Transformations
In classic inf theory, information can not increase through processing, but in real world with alpha zero and syntethic data it does.

-> They resolve by: The classical rule only holds true if the observer has unlimited computation.

Prove via PRG, 
$$H_{\text{Poly}}(G(U_k)) - H_{\text{Poly}}(U_k) \approx n - k$$
-> for a polynomial-time observer, the time-bounded entropy increases dramatically with a one-way function

-> This gives us a crucial rule for Synthetic Data: If you want a deterministic algorithm to generate valuable, interesting data, the function you use must not have a simple, efficiently computable inverse. (then it acts as an information generator for a bounded observer.)

Also example with celluar automata, simple rules are learned instantly, chaotic rules are never learned, but with rule 54, which produces a mix of chaotic noise interspersed with complex, interacting localized structures (gliders, walls, and patterns) loss decreases steady with compute

## Paradox 2: Information Content is Independent of Factorization
In classical information theory, information content is completely independent of how you factor (slice) a dataset.
-> real world contraticts that though, i.e. with text in normal direction being easier to model

* Again resolved by a bounded observer 
If $f$ is a one-way function, $X$ is a secret seed, and $Y = f(X)$ is the scrambled output, a polynomial-time observer faces a strict informational gap:$$H_{\text{Poly}}(X \mid Y) + H_{\text{Poly}}(Y) > H_{\text{Poly}}(Y \mid X) + H_{\text{Poly}}(X) + \omega(\log n)$$
(left side requires inverting a one-way function)

* Empirically they show this with cellular automata rule 30 (which is believed to be a one way function) where the forward converges quickly to the true shannon entropy baseline, but predicting the other way around has a gap

* Also for chess, predicting final state from moves is easier that predicting moves from final state. They theorize that in the later the model needs to develop a richer understanding of math

* I think they don't differentatie between one-way permutations and one-way functions. First is bijective, sencond not. But for the first the initial condition can be found, for the second all possible combintations to get to final state can be found

* Main point: The asymmetry through one-way functions creates epiplexity

## Paradox 3: Likelihood Modeling is Merely Distribution Matching

Common belief: from a particular training distribution, we can at best hope to match the data generating process
-> So in that view from human data no superhuman performance can be learned

"Here we provide two classes of phenomena that seem to contradict
this viewpoint: induction, and emergence. In both cases, restricting the compute available to AI models leads them to extract more structural information than what is required for implementing the generating process itself."

### Induction
They want to show: predicting data requires building complex, inverse-logic neural circuits that were completely absent from the data-generating process itself.

* Murder mystery analogy. Predictior needs to figure out the murderer, the generator (author) does not

To formalize: 
* random variable Z, masking function m(Z) hides h bits of information, a function f(Z) transforms the variable, then final dataset with pairs Y= (m(Z), f(Z))

* Case one: hard induction. he model is given a partial state of Rule 30 where $h$ bits are completely missing. The Task: The model must predict the output $f(Z)$. Because Rule 30 behaves like a cryptographic one-way function, there is no clever shortcut.
-> In cases where the rule is simple, "invertible" means that each element is just a simple function of its neighbours (I believe) and thus the hidden elements can easily be calculated
-> so instead for the hard rule the model resolves to try every combination and then run F

More generally:  For a hard rule, all elements are entangled, thus you can not solve just one without also looking at all the others. For a easy rule they are not entangled so can be calculated seperatly
(I think)

* Case 2: Markov chain, with a 8 x 8 transition probability matrix Z. f generates a text sequence with that matrix. Model gets 
$$\text{Sequence} = [\underbrace{m(Z)}_{\text{The Prompt}}, \underbrace{f(Z)}_{\text{The Text Sequence}}]$$
To solve model needs two strategies:
1. Deduction circuit: When the model encounters a symbol from the $V-h$ visible columns (like symbol A), it doesn't need to guess. It looks back at the prompt $m(Z)$ in its context window, finds the exact probability row for A, and copies it
2. Induction circuit (In-Context Learning): When the model encounters a symbol from the $h$ hidden columns (like symbol G), it looks at the prompt and sees a blank space. To predict what comes after G, it must look at the actual text sequence $f(Z)$ generated so far.If it notices that every time G appeared earlier in the text, it was followed by B, it inducts that the missing matrix value for G must favor B. It uses an Induction Head to dynamically calculate the statistics of the sequence on the fly.

They show that early in learning model only uses the deduction, only later it cracks the induction circuit. 
-> Induction never was part of the data generation process yet still because it is a MLE it has to evaluate how plausible a string X is so it needs induction

The VAE ParallelThe authors point out that this happens in Variational Autoencoders (VAEs) too:To sample a random image from a VAE, you only need the Decoder network. The data-generating process is simple.But to train the VAE or evaluate a likelihood, you are forced to build a massive, highly complex Encoder network whose sole job is to perform induction (approximating the hidden latent variables $P(Z \mid X)$).


-> There is absolutely no mathematical limit to how much larger, more complex, and more intricate an AI model's internal program will be ($S_T$) compared to the tiny program ($G$) that generated its training data.

-> the boundness of the observer is the driver why interesting concepts emerge. If the nn was infinite it could just evaluate every possible future and then pick the best one
(does that mean that in order to get interesting behavior, we need the computational ability of a model to be lower than what it takes to brute force it? maps well to weight decay)
### Emergent phenomena

"One of the most striking counterexamples to the “distribution matching” viewpoint is emergence. Even when a system’s underlying dynamics admit a simple description, an observer with limited computation may need to learn a richer, and seemingly unrelated, set of concepts to predict or explain its behavior."

-> knowing the rules of some cellular automata does not help us in predicting its behavior (as it takes long to evaluate). But we can find shortcuts, at least for parts of it

"observers predicting future states may be required to
learn more than their unbounded counterparts who can execute the full generating process."

* A system is Epiplexity-Emergent if we compare a low-compute observer ($T_1$) and a high-compute observer ($T_2$) predicting its evolution:
- The 1-Step Map: If they only have to predict 1 step into the future, both observers use the exact same simple micro-rule. Their Epiplexities match perfectly:$$S_{T_1}(\Phi(X)) - S_{T_2}(\Phi(X)) = \Theta(1) \quad \text{(A small constant gap)}$$
-The Multi-Step Map: If they have to predict $k$ steps into the future, the high-compute observer keeps using the tiny micro-rule over and over. But the low-compute observer is forced to build a massive internal dictionary of emergent macro-concepts. The gap between their description lengths blows up:$$S_{T_1}(\Phi^k(X)) - S_{T_2}(\Phi^k(X)) = \omega(1) \quad \text{(An infinitely growing gap!)}$$

-> "In words, Φ, X displays emergent phenomena if two observers see equivalent structural complexity in the one step map, but asymptotically more structural complexity in the multistep map for the observer with fewer computational resources"
(Note: They don't consider cases that are really computationally irreducable)

## Epiplexity, Pre-Training, and OOD Generalization

"OOD generalization is fundamentally about how much reusable structure the model acquires, not how well it predicts in-distribution. "
"Two models trained on different corpora can achieve the same
in-distribution loss, yet differ dramatically in their ability to transfer to OOD tasks. This happens because loss captures only the residual unpredictability, corresponding to the time-bounded entropy,
not how much reusable structure the model has internalized to achieve that loss. Epiplexity measuresexactly this missing component"

", Zhang et al. (2024) observed that downstream task performance benefits most from training on type IV ECA (emergent ones) rules over the other ECA rules, "

But: " A model trained on high epiplexity data can learn a lot of structures, but
these structures may or may not be relevant to the particular downstream task of interest."

They show via requential coding that natural data has a lot more epilplexity compared to entropy in comparision to image data. Image data has a lot of unpredictable information.

"In line with our discussion on emergence in Section 5.3.2, it is possible that with significantly more compute much simpler programs can model these natural datasets, such as by directly simulating the basic laws of physics from which the natural world emerges, but the amount of required computation is likely so high that such programs remain inaccessible to any physically realizable observer and we must treat natural data as having high epiplexity for all practical purposes."

Epiplexity Reaches a Hard Ceiling(with inf compute) ($S_\infty$): The total amount of structural information you can extract from a fixed dataset is bounded by the dataset size itself, dictated by the data scaling exponent $\beta$:$$S_\infty(X) = \frac{\beta}{1-\beta} D_0^\beta D^{1-\beta}$$
-> The scale of this ceiling is controlled by $\beta$. A smaller $\beta$ means the model's loss drops very slowly as you add data. Counterintuitively, this means the dataset is incredibly rich and complex—the model is absorbing significantly more structural information per token before it runs out of patterns to learn.

## Related Work
Other notions to extract the relevant info in datasets, i.e. effective complexity, and logical depth.

## Notes
IDK about everything, isn't it basically that emergent behavior via simple rules is complex, and basically generates infinite training data and to learning means finding pockets of reducability?



## Ideas
Shift ARC-AGI training from just training on the data to finding data, that, when trained on, produces good ARC-AGI performance.

* AlphaZero ist interessant, weil es hat ja nur simple regeln basically (sehr related zu Wolfram stuff) und lernt damit aber trotzdem sehr komplexes verhalten. Gibt es einfache regeln, die dann ein Datenset erstellen, mit dem dann NN ARC lernen kann? Well, natürlich der Lösungsalgorithmus, aber ja... :D

* There is something that can be learned by prediction of bag of words, but more when the text is in its proper order. Can we construct a model that only learns from the difference of the two, so it can never rely on the bag of words heuristic? (which does help somewhat but does not generalize...)

* Interesting take Paradox 3: Basically due to the restrictions of the observer, it can not use the simple rules directly, but needs to find pockets of reducability, so emergent structures, to do predictions

* Intersting how the statment about PRG, framed as next token prediction. NN are polynomial time predictors (I guess) and so some stuff they cant figure out, but maybe in languge such big randomness does not exist

* What do NN learn when trained on Game of life stuff? do they discover the correct algo with enough training data? what about giving it to a base model?

* Base model will do a mixture or correct reasoning and spuriius cues/fast reasoning- even when given a lot of data so bascially CoT reasoning. how to elicit reasoning behavior
-> can we, given that we know the correct answer, adjust the llm slithely such that it has the correct rule? what can we learn from what we had to adjust there? can we in this way find out how to generally make the model more reasoning and less spurious cues? find the minimal change maybe

* Higher IQ ppl are just ppl with a higher computatinal power in a sense, so they see less things as random

* Intersting that there is a difference between the rules of chess and the rules to play chess perfectly. Also interesting that the rules to play chess perfectly can be generated from just the rules of chess via RL
Can we have a "rules of language" that then when RLd tells us how to play language perfectly?
IS this related to emergence? simple rules but they kinda just define a space and in that space a competent agent can operate...

* The example with the fluid dynamics: I think it likely coming from the ruliad thinking that anything complex can be described in simple rules thus anything has low sophistication assuming infintie compute. In that sense, shortcuts are essential in learning anything. So maybe shortcut learning isnt a bug, but a feature
-> "random" ist just structutre but with less time to think about it
-> then when do models go wrong? when they try to predict something that they should not yet be able to given theirs training or ttt time. like if at the beginning of training they already try to fit super complex stuff that will fail, because the compure they have at that point will only allow for bad learning?

* From the defintion of P* I noticed a curiosity: It can be that there a multiple P*, i.e. its a line, with a tradeoff between model size and how well a model predicts. Actually a good point also in normal model training, maybe we actually want a worse model prediction wise if that means that we can have a smaller model. 
-> Oh in that view it actually actively hurts to train on all tokens, as then the model will be more complex but it doesnt help us for actually usefull stuff.

* Epipilexity, interesting, how do they measure this computational budget? is it layers? or is it amount of parameters? or amount of GD steps?
-> gemini says from theo viewpoint its the amount of clock cycles, so maps to depth of layers -> ah they explain later in papaer

* Can we construct the inverse scaling laws? (### Existence of Random Variables with High Epiplexity)

* this whole time boundedness, like ... that the model couldn't figure out within the time limit $T$. - it seems to be the core of the issue. the model first learns the easy spurious cues, and then goes on with the error that sill remains. Maybe thats needed to do but maybe there is a different way, i.e. to take some data and really understand/solve it without using the spurious cues as stepping stones...

* Compression of the training process is an interesting concept - Compression is intelligence, but what does it mean to be able to compress the compression algorithm itself (GD?) Can we compress a generalizing algo and then apply it for new arc agi tasks?

-> maybe one does not only need to compress for data, but also compress the compression algorithm in order to find the solution that is best. Like in meditation, not only see the thing but also see the seeing

* Game of 20 questions, can we frame nn training like that? ie it needs to pose questions in order to figure out next token explicitly. mybe good for distillation
Distillation intereting bc no new info, but that is the same as in previous papers that explained it that its still useful

-> maybe its a good idea to have a model make logn predictions, each one restricting the output tokens. maybe at each layer. that way, at each layer information is restricted, and a few layers suffice. then the loss function can be done over all layers. or cot style output log n token for one actual token. so a decision tree is simulated. also then intersting how to construct the decitions, maybe that can also be done as a prediction, so the decision tree itself is also constructuabel. maybe also nice for interpretability.
-> can a arc agi task be posed as a 20 question game? can everything be posed as a 20 question game? just good questions..
we can also predict all arc tokens simulatnously then, only the decisions are then dependent. (that might also be hypergraph related...)

* Requential things seems related to my idea of having an llm that produces syntethic data to then feed another llm to figure out what generalizes

* In Paradox 2 resolving, I am thinking so if ordering matterns and training in the direction of the harder task ist better for getting out more concepts, should we always do that?

* Can we do syntethic data generation by simply testing stratgies and seeing if they are easily reversible? And if yes, its a good syntethic data...

* So, the computational boundeness is a feature? Like if the nn was not bounded, it would learn the simple rule 30. but bc it is bounded it learns about gliders etc etc... interesting

* For induction1, entanglement vs not entanglement maybe that is a good way to think about cellular automata? Also maybe that is related to nn training, can only figure out stuff that is at least somewhat disentangled. Can this even be a hirachy of solutions?
Also related maybe to how the brain is structured, that it is not connected to too many neurons, else one could not compute with it in the bounded compute setting.

* (does that mean that in order to get interesting behavior, we need the computational ability of a model to be lower than what it takes to brute force it? maps well to weight decay)

* Time bascially leads the unfolding of entangled exp functions. If it is untangled time is not needed as everythign can be calculated in parallel. But time is needed for things that interact in between in a entangled and complicated way. Can we classify cellular automata by their entanglement between elements?

* Maybe the point is to find shortcuts that predict it partly, but don't make a spurious prediction. Ie its ok not predict everything, but never predict more than you bargained for kinda

* Maybe like wd walks on the 0-loss line to find lower norm, we could define training to only consider updates that have the same exact loss for all training examples, so we walk the "same loss" manifold
-> Generally we need to restrict the searchspace such that GD converges to a generalizing answer...  maybe think what such a searchspace can look like
Larger batches naturally lead to waling this line? maybe see if with larger batchsize the loss of that batch after the step is less spread out.
This idea better with ES, not backprop...

* Image data has less epiplexity due to unpredictablity of exact pixel. 1. how does this work when we only predict classes and not the next image? also for next image can we adjust such that we only predict the stuff that we also can predict?---

* Interesting discussion with gemini regarding: here in the large compute we assume loops. But inf wide nn are also universal function approximators. But gemini says they so by having a large amount of fine grain detectors. So basically only memorization. so they can solve a function by only memorizing without using any small fundamental rules or any structures. So its like a third kind. 
-> I wonder, can we restrict a transformers possible patterns to be only like a few and instead loop them? So like TRM but instead we let model only have a few patterns, like akin to a lookup of 3 possibilites. Ie 3 kernels that only match to smth specific.
## Concepts
* Sender-Receiver Game
* Game of 20 questions, if I already have info about the target I need less questions


# Compute-Optimal LLMs Provably Generalize Better With Scale

They study compute optimal LLMS and provie a thigher generalization bound for them.

The authors prove that the generalization gap can be decomposed into three primary, interpretable components:  
1. Parameters per Token ($\frac{N}{D}$): Under Chinchilla scaling laws, when you scale up a model optimally, the number of parameters ($N$) and the number of training tokens ($D$) grow proportionally. Therefore, this ratio remains constant.
2. Loss Variance: The token-wise variance of the loss function. The paper proves that as models grow larger, this variance decreases—meaning the model's predictions become steadier and less erratic across different text segments.  
3. Quantization Error: The drop in performance when a model is compressed (e.g., from 16-bit floats to a fixed lower bitrate). The paper demonstrates that this error also decreases as models scale.  

Proving is done by: 
To do this mathematically, the authors introduce a quantized (compressed) version of the model, which we can call $h_Q$, while the full-precision model is $h$. They break the gap apart by adding and subtracting the loss of the quantized model:$$\text{Generalization Gap} = \text{True Loss}(h) - \text{Empirical Loss}(h)$$$$\le \underbrace{[\text{True Loss}(h) - \text{True Loss}(h_Q)]}_{\text{Quantization Error (True)}} + \underbrace{[\text{True Loss}(h_Q) - \text{Empirical Loss}(h_Q)]}_{\text{Generalization Gap of Compressed Model}} + \underbrace{[\text{Empirical Loss}(h_Q) - \text{Empirical Loss}(h)]}_{\text{Quantization Error (Empirical)}}$$

By grouping the first and third terms, they isolate the total Quantization Error. Now, they only need to find a bound for the middle term: the generalization gap of a discrete, compressed model.

Through some math they then prove that the gen gap of the discrete model depends only on N, D and Loss variance

* Because the first component (parameters per token) is constant on the compute-optimal frontier, the behavior of the generalization gap is entirely driven by the other two components.Since loss variance and quantization error both drop as the model scales up, the overall generalization gap shrinks. This provides a formal mathematical guarantee: larger compute-optimal models will naturally have smaller generalization gaps.  

* Bounds are not super tight, but they go down with model size as expected, unlike other measures...

introduced novel token-level generalization bounds for LLMs which are able
to accommodate the non-IID nature of the tokens within the training corpus

## Why less loss varianze is better for generalization
* In standard statistics, if you want to know how well a sample average estimates a true population average, the variance of your data points is everything. High variance means the model is highly sensitive to the exact arrangement or minor quirks of the specific training data it saw. It implies that a slight shift in the distribution on the test set could easily trigger those "catastrophic loss" spikes

"A conceptually useful story about the learning process involves the model accommodating predictive subprograms of progressively larger computational depth and complexity"
When a model is small, it relies on shallow heuristics (like simple n-gram statistics), which break easily and cause high loss variance when the text gets complex. As the model scales along the compute-optimal frontier, it develops deeper internal "subprograms" capable of handling complex reasoning and grammar structures smoothly. This structural depth systematically dampens the erratic spikes in token-wise loss, forcing the overall variance down and anchoring the model's generalization capabilities.

### Proof
the authors model the sequence of token losses as a martingale difference sequence. This is a statistical framework where your next prediction error depends on everything you've learned from the history up to that point.

They think of the LLM’s internal parameters not as a giant soup of numbers, but as a massive library of algorithmic subprograms (or circuits).

When an LLM reads a text sequence, it is constantly routing the tokens through these internal subprograms to make its next-token prediction.

A small model has a highly limited set of subprograms. As it reads a sequence of tokens, it constantly finds itself in situations where it doesn't have a subprogram suited for the text. -> Loss spikes

The authors tie this directly into their Freedman-type martingale inequality.Freedman’s inequality dictates that the probability of a model’s empirical loss deviating wildly from its true expected loss is tightly bounded by the sum of its conditional variances across the sequence:$$\sum_{t=1}^D \text{Var}(L_t \mid \text{History}_{t-1})$$


## Why better quantization is better for generalization
* If a model has a low quantization error, it means its weights are highly resilient to rounding. The core logic of the model doesn't depend on hyper-precise, brittle weight values (which is a hallmark of overfitting).

* They mathematically evaluated the rate at which an LLM absorbs unique information from a dataset relative to its physical size ($N$) on the Chinchilla compute-optimal frontier.They proved that the model’s effective information content grows sublinearly (slower) compared to the raw number of parameters.Because capacity grows much faster than information density, the parameters in a giant model become mathematically redundant and smoothly distributed.

## Memorization vs. Reasoning
They train transformer on normal text vs scrambeld text. They show that when they quant those two networks, the acc of the normal model is higher, indicating that it learned subprograms that can be compressed.

## Ideas

* Can we train a model that only lets GD search where the loss variance is 0 and the quant error is low?

* Can we think of a model as a router for different algorithms, so at every token the most fitting algorithm is chosen?

# Non-Vacuous Generalization Bounds for Large Language Models

The Core Problem: Do LLMs Learn or Just Memorize?

This paper provides the first non-vacuous generalization bounds for pretrained LLMs. In plain terms, the authors mathematically prove that LLMs genuinely generalize to unseen data beyond what they have memorized.

## 3. The Three Main Challenges & How They Solved Them
- Challenge A: Unbounded Loss FunctionsLLMs are evaluated using negative log-likelihood (NLL) loss (or cross-entropy) for next-token prediction. Because a model could theoretically assign a probability of 0 to a correct token, the loss can approach infinity (it is unbounded). Most classical statistical learning theories require the loss to be bounded (e.g., between 0 and 1).  The Solution: The authors introduced a prediction smoothing technique. By mixing the model’s predictions with a uniform distribution (adding a tiny bit of noise), they successfully capped the maximum possible loss, enabling the application of PAC-Bayesian and compression-based bound frameworks without destroying the model's actual performance

- Challenge B: Training on Massive Datasets is Slow to Compute.
The Solution: The authors derived a subsampling-based bound. They mathematically proved that you can calculate the bound using just a randomly sampled subset of the data while maintaining strict mathematical validity.

- Challenge C: Too Many Parameters
Compression-based generalization bounds dictate that a model's description (its size in bits) must be significantly smaller than the size of the dataset. Because LLMs have hundreds of millions or billions of parameters, compressing them enough to satisfy this rule
The Solution: They invented SubLoRA. This is a novel, low-dimensional nonlinear parameterization method that combines LoRA with linear subspace training. SubLoRA forces the model to learn within a tightly constrained, highly compressed mathematical subspace from the very beginning of its training

-> When comparing smaller models to larger models trained under the same SubLoRA conditions, the larger models achieved tighter, lower generalization bounds.
-> This provides empirical proof for a massive theoretical claim: larger neural networks are inherently more efficient at discovering and compressing the underlying structure of data, rather than just using their extra capacity to memorize text.

Also pretrained LLMs achieve significantly tighter generalization bounds than those trained from scrathc

## Sublora
Lora training, but additionally restrict to learn only within a low-dim space such as a line or a plane inside the parameter space

## Notes
So its actually not the neccessarly the largeness of the model in itself that helps generalization, instead the largness of the model helps for a better GD that finds a simpler solution
-> Can one make GD better in other ways? 

* They note that the Sublora trained model produced worse text. Sublora might be better if we allow for more elaborate geometric structures such as the goodfire stuff.


# Function Vectors in Large Language Models
It explores a fundamental mystery of modern AI: How do Large Language Models (LLMs) perform In-Context Learning (ICL)? When you give an LLM a few examples (e.g., Apple -> Red, Banana -> Yellow, Lime -> ?), how does it understand the underlying "function" and apply it to the new input?

the authors discovered that when an LLM reads a prompt containing examples of a specific task, it does not just look back at those examples at the very end. Instead, during the middle layers of the network, a tiny subset of attention heads packages the abstract rule of the task into a compact, single vector.  The authors call this a Function Vector (FV). It acts as an internal macro or command that tells the rest of the model, "Hey, whatever input comes next, apply this specific rule to it."  

Historically, ICL was viewed as a bit of black box—some theorized the model was implicitly fine-tuning itself on the fly, while others thought it was just a massive copying mechanism.

This paper provides direct evidence of functional modularity inside LLMs. It shows that models naturally compress abstract tasks into discrete, steerable vectors. 

## How they found the vectors

2. How Did They Find and Extract Them?To find these vectors, the researchers used a technique called Causal Mediation Analysis.  Pinpointing the Heads: They ran prompts for various tasks (like translating English to French, changing words to plural, or naming country capitals) and carefully patched or blocked different attention heads to see which ones broke the model's ability to do the task. They found that a small, specific set of attention heads in the middle layers are overwhelmingly responsible for moving the "task rule" forward.  

Creating the Vector: Once they identified these "causal heads," they took the average mathematical output of these heads across several examples of a task and combined them into a single vector ($v_t$).  

## Proof by steering
hey gave the LLM a completely blank slate or a natural text sentence with zero examples (e.g., just the word "Germany" or "Laptop").  
Right in the middle layers of processing, they manually injected the extracted Function Vector into the model's hidden states.
The Result: Even though the model had seen no examples, injecting the "Capital" FV caused it to immediately output "Berlin". Injecting the "Plural" FV caused it to output "Laptops".

## Properties of FV

* Layer Specificity: FVs are highly effective when injected into the middle layers of an LLM. However, if you try to inject them into the very late layers, their effect drops to near zero. This indicates that the later layers are reserved for formatting the actual token output, while the middle layers handle the abstract reasoning.

* Vector Algebra (Compositionality): Much like how word embeddings famously allow math (e.g., King - Man + Woman = Queen), the researchers found you can do algebra with functions. For instance, adding two different FVs together can sometimes force the model to execute a complex, multi-step composite task.

* More than just an Output Bias: The researchers checked whether a "Capital" vector just made the model shout out random city names. They found that while the vector does contain information about the output category, it explicitly contains the algorithmic mapping connecting the input to the output.

## Notes
* Interesting. So a model might not look back all the time in a few shot setting and instead create a function. Makes sense actually the function is kinda needed to do computation. So in general prompt-> function -> function application with new input -> related to seeing transformers as fast weight
-> so can we imagine maybe as a transformer having a list of algorithms. And with a a prompt the model decides on a linear combination (or any combination) of those algorithms and then this resulting one is applied.
-> so bascially an llm learns functions and for new input it routes them

# Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models

## Core Concept: The "Engram" Architecture
The Problem: Traditional Transformers waste expensive computational power using deep reasoning layers to reconstruct static facts, local patterns, and idioms.

The Solution: Conditional Memory. It adds a massive lookup table (the "Engram" module) to handle memory storage, leaving the Transformer layers free to focus purely on complex reasoning.

## How It Works (The 3-Step Process)
For every single token the model processes, it executes the following loop:

O(1) Suffix Lookup: The model takes the immediate history (parallel 2-gram and 3-gram windows) and runs them through a fixed mathematical hash function. It instantly pulls a static memory vector out of a massive table.

Context-Aware Gating: The active Transformer layers evaluate the whole sentence context and generate a gate score between 0 and 1.

Information Fusion:

If relevant: The gate opens, and the memory vector is injected directly into the network.

If it's a false alarm (hash collision/wrong context): The gate closes, the memory is discarded, and the model relies on its normal layers.

## Key Takeaways & Breakthroughs
The 25% Sweet Spot: Researchers discovered a U-shaped scaling law for parameters. The optimal model setup allocates roughly 20% to 25% of its sparse budget to Engram memory, and the rest to Mixture-of-Experts (MoE) computing.

Massive Performance Leaps: Under identical computing budgets, adding Engram drastically improved factual knowledge (MMLU), long-context retrieval (Needle in a Haystack jumped from 84% to 97%), and unexpectedly boosted reasoning and coding scores.

Why Reasoning Improved: Offloading static memory retrieval to the Engram early in the network (around Layer 2) "de-noises" the deeper attention heads, giving them more room to handle logic.

Hardware Efficiency: Because the hash is deterministic and relies only on text tokens, the system can asynchronously prefetch memories from cheap Host CPU RAM into the GPU just in time, resulting in massive scaling with zero inference speed penalties.

# A Kernel-Based View of Language Model Fine-Tuning
The Overparameterization Paradox: Classical statistics state that training a model with more parameters than data points leads to massive overfitting, yet LLMs fine-tune beautifully on tiny datasets.

The NTK Microscope: Theorists proved that as a neural network gets infinitely wide (massive parameter count), its training dynamics smooth out. Every weight moves only an infinitesimal amount (Lazy Training), transforming a chaotic training maze into a flat, predictable Neural Tangent Kernel (NTK).

Why LoRA Works (Kernel Taming): * While full fine-tuning lets an LLM warp all dimensions (risking chaotic changes and catastrophic forgetting), LoRA mathematically restricts updates to a tiny, flat, low-rank subspace (a 2D sheet cutting through a 100D sphere).

This forces the model’s internal NTK to shed its high-dimensional noise and become "clean." By taming the kernel, LoRA stabilizes the model, preventing it from forgetting its base knowledge while allowing it to learn new tasks efficiently from a handful of examples.

# Verbalizable Representations Form a Global Workspace in Language Models https://transformer-circuits.pub/2026/workspace/index.html

Ant research on J-Space, i.e. internal concepts that have priviliged access 

" Such thoughts can be articulated out loud, deliberately held in mind, and brought to bear on whatever task the moment demands. This distinction, between our accessible thoughts and our unconscious processing, is perhaps the most striking feature of human cognition."

"Specifically, we observe that language models maintain a privileged set of internal representations, available for report, modulation, and flexible internal reasoning, atop a much larger volume of automatic processing. We identify these representations using a new interpretability technique, which surfaces the concepts a model is poised to verbalize at any point in its processing."

"Thus, our question becomes: within LLMs’ repertoire of vector representations, is there a privileged subset that plays a computational role analogous to the global workspace?"

We identified them by searching for representations satisfying the first property, namely those that are verbalizable. We then discovered that, rather surprisingly, they satisfy the others:

* Verbal report. When the model is asked what it is thinking about, it names concepts represented in the workspace. Swapping one active workspace vector for another changes its answer to match.
* Directed modulation. When instructed to hold a concept in mind, or perform mental calculations, the model is capable of activating and computing with workspace vectors, independent of its outputs. In addition, information that is not typically represented in the workspace can be pulled in when the task requires it.
* Internal reasoning. Workspace vectors can be used to represent the value of intermediate computations, when the model chains inferential steps or composes plans, and intervening on them is sufficient to redirect the conclusion.
* Flexible generalization. The same representation serves as a valid argument to many different downstream computations. In other words, a workspace vector lifted from one context and placed in another is correctly operated on by whatever function the new context supplies.
* Selectivity. The workspace comprises a small subset of the total representational content of the model’s activations. It is required for only a fraction of the model’s behavior, and in particular is not involved in pervasive, routine processing like text parsing or grammatical fluency.

## The Jacobian Lens and the J-space

Designed to identify internal representations that are readily available for verbal report.

"For each token in the model’s vocabulary, the Jacobian lens identifies a vector representation that encodes the potential for the model to verbalize that token in the future."

"Concretely, it computes, for each layer, the average linearized effect of an activation on the model's likelihood of producing a particular token (now or in the future), averaging over a large corpus of contexts (see Methods for details). The averaging step is key, as it distinguishes representations that are verbalizable—poised to be spoken about, should the occasion arise—from those that merely happen to be verbalized in one particular context."

* Workspace is only in the middle layers
* Workspace is quite small and only a few concepts active at one time
* Broadcast format: J-lens vectors compose with many upstream output weights and downstream input weights

"Our findings suggest that the J-space achieves many of the functional properties of the global workspace in the brain, while sharing only some of its architectural properties"

* counterfactual reflection training, which seeks to implant a set of ethical behavioral principles into the model’s workspace in relevant contexts, by training it to articulate those principles if it were interrupted and asked to reflect 
We find that this training measurably improves model behavior in the original, uninterrupted contexts, despite no direct training of the ethical behavior taking place. And indeed we find that, after training, the J-space in these contexts is populated with concepts related to the reflections (ethical, honest, integrity), 

## Methods

The basic idea is to characterize an intermediate activation vector by its first-order causal effect on the model's outputs, over a broad distribution of potential contexts.

So, the idea is to find the tokens that are verbalizable by checking if hidden state h is there, what causally can come as output token. 

To a first order, this is a linear relationship captured by the Jacobian between hidden state and output state. Basically it measures what happens to the output state if the hidden state is slightly changed, so how sensitive it is. High sensitivity means that the output token is likely to be verbalized given the hidden vector.

This is done as an average over a diverse dataset, to distinguish between particularities of a prompt vs what actually can be said over the whole pretrain distributon.

This gives the Averaged Jacobian, which they call the Lens. For a new hidden activation this lens can be applied by multiplication, effectively skipping all later layers and replacing with the linear approximation of the Jacobian. https://transformer-circuits.pub/2026/workspace/png/img_1b62b10ab235e6e7.png

### More on the Jacobian

I was used to seeing backprop as "to make loss smaller, we need to adjust weights like that". But a better view is to say "how small change in weights change the loss". So backprop does not need a loss function it can also just work on the raw output tokens. So it captures how sensitive a element in the output vector is to changes in the input vector. 
Basically instead of $$\frac{\partial \text{Loss}}{\partial W} = \frac{\partial \text{Loss}}{\partial h_{\text{final}}} \times \frac{\partial h_{\text{final}}}{\partial h_{\ell+1}} \times \frac{\partial h_{\ell+1}}{\partial h_{\ell}} \times \dots \times \frac{\partial h_{\text{layer}}}{\partial W}$$

we do
In the Jacobian Lens, we stop short and change the starting point. We strip away the Loss derivative ($\frac{\partial \text{Loss}}{\partial h_{\text{final}}}$) and the weight derivative at the end:$$\frac{\partial h_{\text{final}}}{\partial h_\ell} = \frac{\partial h_{\text{final}}}{\partial h_{\ell+1}} \times \frac{\partial h_{\ell+1}}{\partial h_{\ell}}$$

## Comparison with linear probes/correlation
I was thinking: Ok they want a method to determine which concepts in a hidden vector are verbalizable. They have a dataset of prompts. Why not just collect hidden vector - output vector combinations and then train a model with that/calculate correlation.

Well as it turns out the reasoning is that correlation/nn only captures correlation, and the Jacobian is the average local sensitivity and thereby its a direct statement about the model itself and not a statistical inference. 

However, for gaussian inputs those to measures are the same: (Steins Lemma)
Cov(x, y) = σₓ² · E[f′(x)]
Rearranged, this means the ordinary least-squares regression slope of y on x (which is what correlation is really encoding, once you rescale by σ_y/σₓ) equals the average gradient of f, regardless of how nonlinear f is:

(The Jacobian measures literal mathematical dependency via the chain rule. It asks: "If I physically wiggle this specific hidden coordinate right now, does that force the final layer to move?")

## Notes
* One characteristic of a workspace representation is generalization, i.e. ". The same representation serves as a valid argument to many different downstream computations". In ARC, can we identify concepts and then check if they are used in all 3 examples? by moving them around? and if not, its not a good representation. 
-> in a causal way. Ie the function of the llm should stay the same between examples only the input should change...
-> now that I think of it: transformers are fast weight generators. for our arc example however we should require that for one layer all tokens should apply the same algorithmic step (or no step) and also the same algo step for each example. can we force the architecture to do so?

-> for each layer, insert a token that identifies the layer and the first layer in the group only has access to that token so it can only chose an algo based on that and not on the input. that is the algo chooser then and then after the input can be fed.


* Do I understand correctly that this reflection training in itself leads to ethical behavior? Like maybe also in humans?

* A model is correlational, a derivate is causal? can we use that in nn desgin? design a derivative or smth? or just stuff taht comes directly from a trained model without further tuning?

* Does it address our specific shortcoming (cancellation under non-monotonicity, the x² example)? Not directly, as far as this excerpt shows. The paper doesn't discuss variance-of-gradient, curvature terms, or higher-order Stein-type corrections — it stays in the "first-order, averaged" regime we identified as blind to non-monotonic effects. A concept whose effect on the output flips sign across contexts (helps verbalization in some contexts, hurts it in others) could plausibly average toward zero in J_ℓ even if it's causally important in each individual context — the same cancellation pathology we described for f(x)=x².

* Okay so I was wondering what would happen if a model is trained only with this Jacobian approach and not a loss, so just in the gradient leaving the last multiplcation away. Gemini says: 
Scenario A: Maximizing the Jacobian (The Exploding Whisper)If you train the model to maximize $\frac{\partial h_{\text{final}}}{\partial h_\ell}$, you are telling the model: "Make the final layer hyper-sensitive to the early layers." * The Result: The model’s weights will blow up to infinity. A tiny microscopic flutter of static noise at Layer 1 will be amplified exponentially by every layer until the final layer outputs completely chaotic, high-voltage gibberish.Scenario B: Minimizing the Jacobian (The Silent Wall)If you train the model to minimize the Jacobian (make it 0), you are telling the model: "Make sure early layers have absolutely zero effect on later layers."The Result: The weights will all shrink to zero. Every layer will become a completely insulated wall. You could feed the model the most beautiful poetry, and by Layer 2, the signal would be entirely dead.

-> Now this seems very interesting. Seems related to weight decay. Seems related to optimal processing, ie computation at the edge of chaos. Can we use this in training, comboined with normal loss training, so it doesnt go in either direction and stays at the edge of chaos bascially? Or maybe that one input element is only allowed to be senstive for one output element or smth. we could try different methods to see if one works as a good regualrizer.
-> Ok it seems this is called Jacobian Regularization haah


# GrokAlign: Geometric Characterisation and Acceleration of Grokking
(In the introduction, many intersting papers are linked)

Prior research argued that grokking happens when a network transitions from a simple "linear/lazy learning" phase to a "feature learning" phase, reshaping its boundary structures.  This paper provides a new mathematical explanation by analyzing the network's Jacobian matrix (a matrix of all first-order partial derivatives of the network's outputs with respect to its inputs/parameters). Specifically:  The authors show that grokking is structurally embedded within the network’s Jacobian.  They prove that if you align the network's Jacobians with the structure of the training data (maximizing their cosine similarity), the network is guaranteed to achieve grokking, provided the Jacobian matrix satisfies a low-rank assumption (meaning it focuses on a compact set of core features).  

## Jacobian Regularisation Explains Grokking

Network $\text{argmax}(f(x))$

The Jacobian matrix, denoted as $J_x(f)$, represents the sensitivity of the network's outputs relative to its inputs at a specific point $x$. It captures how the network's predictions change if you slightly nudge the input.

* Definition 1: Jacobian-Aligned
The authors define a network as being Jacobian-aligned at a point $x$ if its Jacobian matrix can be written as an outer product of two vectors:$$J_x(f) = c x^\top$$
->Usually, a Jacobian is a complex matrix where every output class responds differently to every feature of the input. If a network is Jacobian-aligned, it means the entire matrix collapses into a highly simplified, structured form. The network's outputs become uniformly sensitive only to the specific directions defined by the input data vector $x$ itself.

* To prove their theories, the authors look at continuous piecewise affine networks. This category includes most modern architectures that use ReLU
-> A ReLU network doesn't create a perfectly smooth, curved mathematical surface. Instead, it chops up the high-dimensional input space into a vast collection of flat, convex distinct regions (polytopes). Inside any single region $\omega_x$, the network behaves exactly like a simple linear equation:$$f(x) = A_{\omega_x}x + B_{\omega_x}$$
Where:$A_{\omega_x}$ is a matrix acting as the local weights for that region.$B_{\omega_x}$ is the bias vector for that region.Crucially, inside this region, the Jacobian is exactly equal to this matrix: $J_x(f) = A_{\omega_x}$.3 (because its like taking the derivative of a linear function)


* Theorem 2: if you train a network to minimize a standard loss function (like cross-entropy or mean-squared error) under two strict constraints:The Frobenius norm (the total magnitude) of the Jacobian is bounded: $\|J_{x_p}(f)\|_F^2 \le \alpha$The local bias term is zero: $B_{\omega_x} = 0$Then the mathematical solution that minimizes the loss is guaranteed to be Jacobian-aligned ($J_{x_p}(f) = c x^\top$).
(so basically if you are training with wd, thus restricting the norm, eventually you will end up jacobian aligned)

* Theorem 3: if the local weight matrix $A_{\omega_x}$ is rank-one (meaning it is highly compressed and focuses entirely on a single subspace of features) and the bias is zero, then setting $A_{\omega_x} = c x^\top$ yields the most robust local mapping possible against $\ell_2$ input perturbations.
The Catch: In practice, standard deep network training naturally biases models toward low-rank weight matrices over time.
The Conclusion: Because networks naturally drift toward low-rank states, achieving true robustness requires the network to achieve Jacobian alignment.

### Explanation 1:
For those kinds of functions, for a area around x, the function is linear. So a Hyperplane.

Normal nn before alignment is complex. Normally, a network's Jacobian is a massive table of arbitrary numbers.If you feed an input $x$ into a regular network, its internal weights can take a messy path to get to the correct output. The network can say: "I will use feature #1 to classify point A, feature #2 to classify point B, and feature #3 to classify point C."This is memorization.

But when its Jacobian aligned: Because the local linear region $A_{\omega_x}$ is exactly equal to the Jacobian ($A_{\omega_x} = c x^\top$) (simple math, ableitung), look at what happens when the network computes its output for that region $f(x) = A_{\omega_x}x$: $$f(x) = (c x^\top) x$$ $$f(x) = c (x^\top x)$$ $$f(x) = c \Vert{}x\Vert{}^2$$

### Explanation 2:
So: a nn is a patchwork of linear functions. (he number of these patches scales as $O\left(\left(\frac{N}{L}\right)^{Ld}\right)$, where $N$ is the number of neurons, $L$ is the depth, and $d$ is the input dimension.)

In the non-jacobian algined, A and B is arbitray and as such it could be that inside the linear patch its very suspectible to small changes in x.

(And we get jac algined because we restrict the norm via i.e. weight decay and thus theorem 2 says we get to jac alignment)

However, when we are in the jac-aligned phase, $A_{\omega_x} = cx^\top$ and thus $$f(x) = (cx^\top)x = c\|x\|^2$$. Small changes here in x barely change the output. We bascially just defined c as the "vocab token" for this linear patch.

The c's must kind match at their border region, because a nn is a continous function.
Also, when we drive the norm down, we probaly can't have a distinct c for every linear patch, instead we only have a "small" set of c. Those c are then basically "pseudo vocab tokens" that are then decoded to real tokens via argmax. So in this view a transformer doesn't predict a arbitrary prob distribution for a token, it predicts a "pseudo token". 

Ok now, with theorem 3 we then know that jac aligned is the most robust to pertubations. Now they argue that is then also the grokking phase. They don't provide a theoretical argument as far as I understand, though have empirical evidence.

-> So GrokAlign is used to lower/bound the Jacobian norm
## 3 The Centroid Alignment Perspective

Calculating the full Jacobian matrix ($J_x$) for every single training point during a model's run is computationally brutal. It destroys training speed. To fix this, the authors introduce Centroids, a geometric simplification that summarizes the Jacobian into a single vector.

* As we established, a ReLU network chops the input space into millions of local linear regions (convex polytopes). Every patch has its own local Jacobian matrix $J_x(f)$.Theorem 4 introduces a way to condense that whole matrix into a single vector, called the centroid ($\mu_x$), by multiplying the transposed Jacobian by a vector of all ones ($1$):$$\mu_x = (J_x(f))^\top \mathbf{1}$$
-> Basically it summing up the rows in the Jacobian, so we don't know how every single output class behaves with changes in the input, but only the average output class.
-> This can be calculated efficiently with the Jacobian-Vector-Product JVP
-> Summarization mechanism in a ay that has an "elegant geometric interpretation"

* Defintion 5: Centroid-Aligned: A network is centroid-aligned at a point $x$ if its centroid vector points in the exact same direction as the input vector itself:$$\mu_x = c x \quad \text{(for some scalar } c\text{)}$$

* Proposition 6: A Jacobian-aligned deep network is centroid-aligned, If a network is Jacobian-aligned ($J_x = cx^\top$), it is guaranteed to be centroid-aligned. Centroid alignment is just a slightly relaxed, easier-to-calculate version of Jacobian alignment.

* Link to Grokking via "Region Migration": 
Before Grokking: The network places a dense cluster of tiny, chaotic linear patches directly on top of the training data points so it can memorize them.
During Grokking: The network forces these linear regions to "migrate" away from the data points and stack up neatly along the decision boundaries instead.

So memorization:To memorize a dataset, the network creates a dense, chaotic honeycomb of tiny tiles directly on top of the training data points.Each training data point gets its own tiny custom tile.Inside that tile, the local matrix $A$ is engineered to force a correct output for that specific point, but it tilts crazily. The local Jacobian is messy and high-rank.This is why it lacks robustness: if you nudge the input slightly, it slips out of that custom tile into a neighboring tile where the matrix $A$ points in a completely random direction.

----
A centroid is literally the anchor point in the power diagram view

If a network is centroid aligned, this means that the geometric anchor point of the tile points in the exact same directional ray as the data point $x$ itself. because $\mu_x = cx$ here c is a scalar.

If you track the centroid vectors over time and see them suddenly lock into alignment with the data vectors, you are watching Region Migration happen in real-time. It tells you the tiles have successfully cleared away from the data points and expanded.

Why is it the case that the centroids need to be aligned to the data vectors for grokking to occur? 
-> Well I dont complety get it. however, i believe the story is that a stright line between two data points is the boundry with the lowest norm. And the boundtry is straight, if the centroids are a multiple of the data points, or else the line will at least be slightly off or ragged.


### Voronoi and Power Diagrams
Standard Voronoi Diagram: You have a set of seed points. Every spot in the space belongs to the closest seed point based on standard Euclidean distance. 

Power Diagram (Laguerre-Voronoi): This is a Voronoi diagram where the seeds have different "weights" or sizes. The boundary between two cells is determined not just by proximity, but by a "power distance" that factors in these weights. The boundaries are still straight lines (hyperplanes), but they get shifted based on the weights.

Relationship to Relu:
A single ReLU function is defined as:$$f(x) = \max(0, x)$$ (one side linear, the other 0)
The boundary where this switch happens ($w^T x + b = 0$) is a hyperplane (a line in 2D, a flat plane in 3D, etc.).When you have a whole layer of ReLU neurons, you are essentially dropping a bunch of these flat hyperplanes into your input space

Mathematically, it has been shown that the boundaries created by a layer of a ReLU network can be mapped identically to the boundaries of a Power Diagram.

A relu network is a piecewise linear function. For those piecewise linear functions, the power diagram can be extracted with 
If you know the local linear function $F(x) = A_i x + B_i$ for a region, its corresponding Power Diagram components are:Seed Position ($p_i$): $\frac{1}{2} A_i$Seed Weight ($w_i$): $B_i + \frac{1}{4} \|A_i\|^2$
This must be done per region.

also: The tiles are not uniform. Wherever the weights are changing rapidly or are very large, the hyperplanes pack tightly together, creating an ultra-dense cluster of microscopic tiles. Where weights are simple or uniform, the tiles stretch out into massive, yawning expanses.

## Relationship to NTK

Let’s look at the "Smoking Gun" equation again:$$\partial_t (\langle x, \mu_x \rangle) = \eta \frac{1}{m} \sum_{p=1}^m \Theta(x, x_p) m_{x_p}$$To understand why this relates to migration, we have to look at what the Neural Tangent Kernel (NTK), $\Theta(x, x_p)$, actually measures. The NTK measures representational similarity. It asks: "If I update the network's weights to learn about training point $x_p$, how much does the network's output change at point $x$?"

So I think:
during ealry training if we take a trainstep, the prediction of other points in data space dont change much if we adapt the label for xp?
because we have those tiny tiles and basically we only adapt the tiny tile? and later when we have big tiles and we change a label of a tile this affects a large area in input space? 

Ah so this is an explanation for generalization: during early training with the many small tiles, adapating one tile, so one training example, does not change any other inputs, so also not those we want to generalize to.
And in other ways, if we are in the grokking regime, a change in one tile also has changes to all related inputs.

## GrokAlign algorithm

$$\text{Loss}_{\text{GrokAlign}} = \text{Loss}_{\text{Task}} + \lambda \left( \Vert{}J_x\Vert{}_F^2 + \Vert{}b_x\Vert{}_2^2 \right)$$

Step A: Erasing the Biases ($\Vert{}b_x\Vert{}_2^2 \to 0$)

Step B: Squeezing the Slopes ($\Vert{}J_x\Vert{}_F^2 \to 0$)By penalizing the total energy of the Jacobian matrix, the optimizer is blocked from creating a high-rank, chaotic "broken glass" landscape of rapidly changing slopes from tile to tile.

-> forces directly to jacobian alignment
-> Under normal training conditions, a network spends thousands of epochs wandering blindly in the dark before its weight configurations happen to compress enough to trigger region migration and generalization.
Because GrokAlign directly measures and punishes unaligned geometry at every step, it forces Region Migration to happen almost immediately.


## Notes
* It seems like one could characterize a model training from a different perspective: not to minimize loss, but to minimize the norm given the contraint of low loss. this makes the training seem a bit dumb because norm minmzaation with wd is i think bascially randomly walking around, its not such a directed way like the gradient of the function? maybe there are better ways to find a simple solution given a loss contraint.

* Interesting to see a network as this patching of linear areas. Even more interesting that in the jac aligned, a nn is basically just choosing a vector c for every input (because $$f(x) = c \|x\|^2$$). So it could be seen as if the nn just has a larger vocab and maps each token to one of those "pseudo-vocabs". Then each pseudo-vocab is decoded to the actual vocab via argmax. Wow interesting. This seems relevant to interpretabilty.

* What would happen if we manually restrict the output of each layer to be one of a fixed set of pseudotokens? Then we automatically are in a jac-aligned phase?

* This tiling is nice, I think thought that it is 3-dimensional in the sense that each point in input space is associated with n tiles, so its multiple mosaics stacked

* Also a nice perspective with the tiling: Seeing it as a "zusammenhalt". If we have a generalizing tile, then if we change the label of a tile (or take a hypothetical small step), this changes all datapoints inside this class. With that we can do testing maybe? Like if we do a small step in any direction for a input of class x, we also want the label for all other inputs of class x to change. because if the point is inside a tile, changing it will change the tile basically.


# https://alexzhang13.github.io/blog/2026/mgh/

Proposes that current LLM capability is enough, but they need to be in better harnesses. Basically, they need to be fine-tuned to be able to decompose problems a la RLM. If the process of decomposition and the decomposed parts are all in-distribution, then even if the original problem is out of distribution, the model will generalize.

"the MGH posits that modern LMs are so good yet so expensive to further train, that directly learning the operator to compose LMs is a significantly more efficient strategy for reaching these OOD tasks than continuing to scale current LMs."

## Notes
* Composition as OOD generalization? Divide and conquer

* Ok that is a very interesting thought: We don't need a model to generalize natively, but we need a model that is capabale of decomposing a problem such that it understands it. Divide and conquer basicially, split, conquer, merge.
(underlying philosopy: A single massive problem (or enemy) is far more dangerous than the sum of its parts. As a system or group grows, the number of internal connections and potential energy grows exponentially. By severing those connections, you drastically reduce its overall strength. -> so one is reversing the effects of emergence in a sense) 

* For ARC-AGI, can we do the same? so instead of relying on the llm to do nice decomposition, we explizitly train for it? One llm that extracts part of the grid, another one that then solves it for all examples, then it gets merged back etc. maybe the choosing llm always has to choose all tokens to one of several calculator llms. basically how llms work internally, but explizitly in a discrete way.

* For long context worK: Can we construct a min-max game, where model needs to get maximally good score but only using minimal token amount? it can delete its own tokens, and query from long prompt etc etc, but its "working memory" is tiny. Those kinds of models will also be easier to train as they dont need such massive amounts of context. they can also store stuff in database etc.

# https://alexzhang13.github.io/blog/2026/harness/

"Modern post-training has become a brute-force paradigm of curating ever more environments and ever longer training horizons. In large part, this is because frontier Transformers are still poor at compositional generalization, the ability to solve unseen problems by composing familiar ones."

-> Better generalization via harness
"The primary job of the harness should be to carry a higher-level (than token level) inductive bias that can reduce unfamiliar and complex problems to compositions of simpler ones for the underlying neural network."


"good harness is one that shapes each call to the underlying Transformer so that every observation is locally in-distribution,"

Here they show: "what a model learns through a well-designed harness generalizes across task lengths and across domains far better than training the neural network on its own does."
-> they train a LLM with a RLM harness and show that it generlaizes far better for longer sequences and data shift

## Notes
* Maybe: OOD generalization is not possible in the strict sense. Instead OOD generalization comes from being able to decompose arbitrary problems. Maybe those two are the same...

* I can't just put my finger on that but I have the inutition that larger models effectivly are akin to training with a RLM harness, in that they enable compositinality and thus better generalization.

* Compositinality (being able to do divide and conquer) might be the same as pockets of reducability, just looked at inverse
