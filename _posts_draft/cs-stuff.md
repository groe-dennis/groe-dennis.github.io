# Online learning

## Regret minimization as theoretical foundation of online learning
RT​=t∑​Lt​(wt​)−wmin​t∑​Lt​(w).

So basically Loss is differnt at each step. Regret measures the current loss against the current loss of the hypothetical best weights.

Why not just use 
∑T​Lt​(wt​).
so the cum sum of the losses?
1. With hard problems, the cum will be large and so the number is not comparable between different problems
2. The sum sum is trivially less when we take fewer steps, so thats not good.
-> but that could be solved with proper regularization

But regret is also spritually different. Comes from a setup where one also expects adverserial losses.

-> With this adverserial thinking in place, RM makes no i.i.d assumption, the data distribution can drift over time

* $$\lim_{T \to \infty} \frac{R_T}{T} = 0$$ Is the goal, meaning that regret is sublinear and converges to 0.


## Multi-Armed Bandit (Subset of Online Learning)
* In a Bandit Feedback model, the environment is restrictive. If you select action $a$, you only observe the reward or loss for action $a$
* A Multi-Armed Bandit is a stateless model. Every round is an independent trial.

* Constant Dilemma of Exploration vs. Exploitation

* Regret minimization provides a structured, mathematical way to balance this. Algorithms like Upper Confidence Bound (UCB) use regret bounds to ensure you explore just enough to not miss out on optimal choices, but not so much that you waste resources.

## UCB in Multi-armed bandit

* If you don't know the true mean reward ($\mu_i$) of the arms, a naive approach is to calculate the empirical average reward ($\hat{\mu}_i$) for each arm based on past trials and just pull the one with the highest average.This fails. If you pull a spectacular arm once, get a fluky reward of 0, its empirical average becomes 0. 
-> A naive algorithm might ignore it forever.

Instead, use UCB1 which tells you to choose the arm that maximizes
$$\text{UCB Score}(i) = \hat{\mu}_i(t) + \sqrt{\frac{2 \ln t}{N_i(t)}}$$

-> $\hat{\mu}_i(t)$: The Exploitation Term. The empirical mean reward of arm $i$ up to time $t$
-> $\sqrt{\frac{2 \ln t}{N_i(t)}}$: The Exploration Term (The Uncertainty Bound). (log of the number of times played so far divided by the number of times arm $i$ has been pulled specifically)

-> If a arm is ignored, its uncerainty grows, if you pull it enough time the uncertainty shrinks and it basically goes to the average

(the specific form with the square root comes from the Hoeffding's Inequality: Hoeffding's inequality proves that the probability of the true mean $\mu_i$ being larger than your empirical estimate $\hat{\mu}_i$ by some padding $\epsilon$ decreases exponentially with the number of samples:)


# Feature Selection im Maschinellen Lernen

oft Tausende von potenziellen Einflussfaktoren (Features). Viele davon sind aber redundant oder korrelieren miteinander.

Wenn du ein Modell mit allen Faktoren fütterst, neigt es zu Overfitting (es lernt auswendig, statt zu verstehen) und wird extrem langsam.

Nutzt man hier ein Greedy-Verfahren (oft Forward Selection genannt), sucht man das wichtigste Feature heraus.

Durch die Orthogonalisierung zieht man den Effekt dieses ersten Features von allen anderen ab. Das nächste gewählte Feature bringt also garantiert neues, unabhängiges Wissen ins Modell

## Ideas
Wenn man nur unabhängige feature in ein nn bringt, gibt es dann die perfekte, generalisierende Lösung aus?

Scheinbar ist da was dran. Also sollte jeder feature vektor im stream maximal orthogonal sein oder so

#

"Dann ist ein Gradientenschritt ungefähr wie:
text
 
     
 
 
1
2
Nimm die Richtung, die den Loss am meisten reduziert,
ohne frühere Fortschritte stark kaputtzumachen.
 
 

Das ähnelt Greedy bei orthogonalen Features.

In einem linearen Modell mit orthogonalen Features ist Greedy tatsächlich ideal:
text
 
     
 
 
1
y ≈ Σ_i w_i x_i
 
 

Wenn die x_i orthogonal sind, dann kann man jedes Gewicht unabhängig bestimmen. Es gibt keine Interferenz.

Bei neuronalen Netzen sind Features nicht exakt orthogonal, aber bei großer Breite kann ungefähr gelten:
text
 
     
 
 
1
2
3
4
Feature-Interferenz ↓
Gradientenrichtungen werden unabhängiger
Conditioning verbessert sich
Loss-Landschaft wird lokaler konvexer
 
 

Dann wird Gradient Descent zuverlässiger."


# Sparse approximation

Man sucht einen Vektor $x$, der das Gleichungssystem $y \approx D x$ löst, unter der Bedingung, dass die Anzahl der Nicht-Null-Einträge in $x$ (die sogenannte $L_0$-Pseudo-Norm) so klein wie möglich ist.

$$y \approx \sum_{i} x_i d_i$$

Also quasi Du möchtest $y$ als Linearkombination aus so wenigen Atomen wie möglich darstellen.

## Lösung 1, greedy, Matching Pursuit

Der Greedy-Ansatz (Matching Pursuit)Ein Standard-Greedy-Algorithmus geht so vor:
* Suchen: Finde den Vektor $d_i$ aus dem Wörterbuch, der die größte Ähnlichkeit (das größte Skalarprodukt) mit deinem aktuellen Signal $y$ hat.
* Abziehen: Ziehe den projizierten Anteil dieses Vektors von $y$ ab. Was übrig bleibt, ist der Rest (Residuum $r$).
* Wiederholen: Nimm das Residuum $r$ als dein neues Signal und suche den nächsten Vektoren.


Problem hier ohne Orthogonalität:
Wenn die Vektoren im Wörterbuch nicht orthogonal sind, kann es passieren, dass der Algorithmus im nächsten Schritt wieder einen Vektor wählt, der dem ersten sehr ähnlich ist: 

* Kann immer wieder ähnliche Atome auswählen, die kleine Fehler in früheren Atomen ausbesser, läuft damit quasi im Zick-Zack
* Frühe "schlechte Entscheidungen" können nicht korrigiert werden
* 

Lösung ist Orthogonal Matching Pursuit

## Orthogonal Matching Pursuit

MP: „Ich füge einfach das nächste passende Atom hinzu.“
OMP: „Ich füge das nächste passende Atom hinzu und rechne dann die ganze Mischung neu aus.“


1. Wieder atom wählen wie bei MP, der das residum minimiert, mittels dot-product
2. Dann schaut man sich aber die bisher genutze Atom-Menge an, und mit dem neuen Atom dazu lösung man Least-aquares Problem um die Koeffizienten neu zu optimieren. Quasi ein Korrekturschritt (Atome bleiben gleich nur koeffizienten ändern sich)
3. Dann wieder das Residuum berechnen und das nächste Atom auswählen

Das neue Residum ist dann immer orthogonal zu allen bisherigen Atomen. (Wenn Atom 1 und 2 zB ein Ebene aufspannen, wird Residuum vertikal darauf stehen. Deswegen werden dann keine Atome mehr gewählt die auf der gleichen/ähnlichen Ebene sind wie Atom 1 und 2)



-> Wenn dein gesamtes Ausgangs-Wörterbuch bereits aus paarweise orthogonalen Vektoren besteht, ist MP greedy schon optimal!
-> Das ist zB bei JPEG so, dort sind die Atome so vordefiniert, dass sie schon orthogonal liegen. Dort sind Atome für menschliche Augen relevante Vekoren wie Farbverläufe und Karomuster

Bei MRT hilft OMP weil man nur einen Bruchteil der Daten messen muss, aber OMP trotzdem gutes Bild rekonstruiert.

## Ideas
Kann man das einfach als standard nn training sehen?
Also quasi in jedem gd schritt wird ein gradient drauf addiert. Also ist jeder gd wie ein feature das addiert wird, mit dem ziel das am ende y, also die prediction für das gesammte Datenset approximiert ist

Das ist ein anderer BLick auf gd, nicht durch loss landschaft laufen, sondern gd lernt quasi das nächst wichtigste feature und fügt es hinzu. Das erklärt warum gd an sich optimal ist, also quasi "den besten abstieg" maxcht, es fügt immer das beste feature hinzu

ok jetzt gibt es ja aber OMP, aus MRI research, hilft uns das? Machen große modelle automatisch OMP?

Quasi eigentlich: jedes feature in dem llm ist ein eigenes atom, und wenn man die atome aufaddiert bekommt man intelligenz

* OMP ist interessant. Es ist ein greedy algorithmus, aber in jedem Schritt führt er erstmal noch ein Cleanup der Historie durch. Früheres Error Correction quasi. Macht greedy natürlich nicht direkt optimal aber vlt besser. Bei nn vielleicht erstmal alte gradienten verbessern bevor man einen neuen gradienten aufaddiert?

* Chat sagt dass OMP bei Sparse Coding und Dictionary Learning eingesetzt wird, kann man nutzen um SAEs zu verbessern?

# Projection pursuit
In high-dimensional data, most low-dimensional projections just look like a meaningless, blurry cloud of points. This is actually a mathematical fact as dimensionality grows, most random projections of data tend to look normally distributed (Gaussian).

-> Projection Pursuit tries to find "interesting" projections, so maximally not gaussian

-> Via optimization: proejct into lower subspace, measure its interestingness, optimize for interestingness

Measures of non-gaussianity can we varied, one is "kurtosis" (standard normal has 0, clusters and heavy tails have high kurtosis)

## Relationship to PCA 
PCA maximizes variance, PP maximizes non-gaussian

## Relationship to the Curse of Dimensionality
In high dimensions, all points become roughly equidistant from one another.
-> Traditional distance-based algorithms (like $k$-Nearest Neighbors or density estimation) completely break down because the concept of "nearness" loses its meaning.

-> PP bypasses the high-dim problems by doing all calculations in low-dim

## Relationship to other algorithms 
Independent Component Analysis (ICA): Essentially a fast, specialized form of Projection Pursuit used heavily in signal processing

Projection Pursuit Regression (PPR): An additive model that models a response variable by summing smooth functions of projection pursuit directions. It was a direct precursor to modern neural networks.


## Ideas
* With a lot of space, all points are equidistant? Seems related to all is one, given enough spaceiousness

* Can one use a nn for down projection and just use kurtosis as loss? what happens then

* Can one use a nn for down projection and search for all the interesting relationships?

# Projection Pursuit Regression (PPR)

## Ideas
This seems quite like the idea I was having to look at sparse approximation

Actually quite nice to thin

# Independent Component Analysis (ICA)

The Classic Example: The Cocktail Party Problem

Microphone 1: $0.6 \times \text{Speaker A} + 0.4 \times \text{Speaker B}$Microphone 2: $0.2 \times \text{Speaker A} + 0.8 \times \text{Speaker B}$

CA takes these mixed signals and separates them back into the original, pure audio streams of Speaker A and Speaker B.

By the Central Limit Theorem, when you mix independent signals together, the mixture looks more Gaussian than the original signals. ICA exploits this in reverse: it uses Projection Pursuit to rotate the mixed data until the resulting axes are as non-Gaussian as possible. When non-Gaussianity is maximized, the original independent signals cleanly separate.

# Kernel Trick

The Problem: Many datasets are linearly inseparable in their original space (e.g., points arranged in concentric circles in 2D). They require mapping into a higher-dimensional space ($\Phi(x)$) where a linear boundary (a hyperplane) can separate them.The "Trick": Manually transforming data into high-dimensional spaces is computationally expensive or impossible (e.g., infinite dimensions). 

A Kernel Function ($K(x, y)$) bypasses this by calculating the similarity score (the dot product) of those points in the higher-dimensional space using only their original, lower-dimensional coordinates. 

## Disadvantages

Quadratic Complexity and overfitting (?)
# Kernels/Linear Attention
Standard Softmax self-attention $\text{Softmax}(QK^T)V$ computes a giant token-to-token similarity matrix, incurring an expensive $O(N^2)$ quadratic bottleneck over long text lengths.The Kernel Fix: Researchers realized Softmax attention is a similarity kernel. By replacing Softmax with explicit, linear kernel functions ($K(q, k) = \phi(q)\phi(k)^T$), they flipped the matrix multiplication order, dropping the complexity to linear $O(N)$.


1. Kernel functions, by trying to be computationally easy, usually drop the exponential Softmax. Instead, they smooth everything out into a broad average.

The Result: Linear attention kernels suffer from "fuzzy vision." Instead of perfectly recalling a specific word, they return a blurred mixture of the target word and the words next to it.

2. The Compression Performance WallIn standard Transformers, as you type a longer prompt, the "KV Cache" grows. The model preserves every single past token explicitly.  In a linear attention kernel, because of how the associative law flips the math, the past is compressed into a fixed-size memory matrix (a hidden state). As the text grows from 1,000 to 32,000 tokens, the size of that memory matrix stays exactly the same.  The Problem: You cannot perfectly fit a 100-page book into a 1-page summary without losing information. As context grows, linear attention hits a "performance wall" where its perplexity (accuracy) plateaus or degrades, whereas standard attention keeps getting smarter the more context it sees.