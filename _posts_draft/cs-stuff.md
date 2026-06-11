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

