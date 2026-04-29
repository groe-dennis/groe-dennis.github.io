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

## Method
* Every mirror symmetry (reflection symmetry) in the loss function forces a structured constraint on the parameters.
* When such a symmetry is present, and gradient noise or wd is strong enough, SGD has a strong tendecy to solutions satisfying Oᵀ θ = 0 (parameters lie in the subspace orthogonal to the symmetry)

* To implement, the most straight-forward method is to multiply a weight matrix with a scalar v

* More advanced, other matrix multiplcations can be done, such as the hadamard product w_i = u_i × v_i, 


* The key property is that the reparameterization is faithful — the model can still represent the exact same functions as before (no loss in expressivity when the symmetry is fully broken), but the training dynamics now have a strong bias toward the structured solution you want.

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