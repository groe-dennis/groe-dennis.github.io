# Idea
Have a vector that represents DNA. Use this vector to construct a big model, maybe task specific. Train the vector. Idea is that this guarantees that the model is able to generalize. 

## Previous approaches

### HyperNEAT
Small Model that you can query for a specific NN position, it returns the weight. So you can like query for weight layer n, input i, output j weight

### Hypernetworks
Models that can generate the weight for other models

### Compressed frequency-domain vector, Evolving Neural Networks in Compressed Weight Space
represent each weight matrix as a short vector of Fourier / Discrete Cosine Transform (DCT) coefficients—just like JPEG compresses an image by keeping only the important low-frequency patterns. Then Run the inverse DCT → reconstruct the full weight matrix

See also https://arxiv.org/pdf/2504.03037
and https://arxiv.org/pdf/2508.08526v1

### Low Rank Factorizations are Indirect Encodings for Deep Neuroevolution
Instead of evolving a full weight matrix they evolve Low Rank Matrices

### Training and Generating Neural Networks in Compressed Weight Space
Represent every weight matrix W not by its millions of entries, but by a tiny vector of DCT coefficients (Discrete Cosine Transform – the same transform JPEG uses for images).
A cheap, fully differentiable decompression step turns those few coefficients back into a full weight matrix on the fly.
This gives fine-grained control over parameter count (you just choose how many DCT coeffs to keep) and makes “fast weight programming” (one net dynamically writing the weights of another) scalable.

# How does evolution solve the DNA->Organism
The egg already has a full diploid genome plus a bunch of “maternal effect” mRNAs. Think of maternal factors as the bootloader on a computer.

The zygote divides. All cells are genetically identical, but they inherit slightly different amounts of maternal factors depending on where they sit. This is the first “position information.”

At a species-specific point, the embryo’s own DNA wakes up. Transcription factors (proteins) start binding to regulatory DNA sequences and turn on the first wave of “zygotic” genes.

DNA → mRNA (transcription) → protein (translation).
Every gene has:
Coding sequence = “build this protein” (the “what”).
Regulatory sequences (enhancers, promoters) = “turn this gene on only if these transcription-factor proteins are present at these concentrations” (the “how” and “when”).

The GRN itself has feedback loops, timers, and mutual repression so that once a cell picks a fate, its daughters tend to stay in that fate

The same logic scales up: signaling centers (organizers) in developing limbs, brain, heart, etc., set up new local gradients and GRN cascades.

-> It’s an executable program that runs locally in every cell, using the same DNA but different inputs (position, neighbors, history) to produce different outputs.

## Modeled as a Boolean Network
Boolean GRN for one cell
n = number of genes (nodes)
f = list of Boolean functions, one per gene
Example: f[0] could be "gene0 = gene1 AND NOT gene2"

initialize state[0..n-1]   # e.g., [1, 0, 0, ...] from maternal factors

for each time step t from 0 to T:
    new_state = empty array of size n
    for each gene i in 0 to n-1:
        inputs = state of the regulators of gene i   # look up which genes regulate i
        new_state[i] = f[i](inputs)                 # evaluate Boolean logic: AND, OR, NOT, etc.
    state = new_state                               # synchronous update: all genes at once
    # Optional: interpret state for cell behavior (e.g., if geneX == 1 → divide, move, etc.)

local state can be modeled as an attention mechansim.
