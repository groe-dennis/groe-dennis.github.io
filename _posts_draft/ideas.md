GEPA but for LLM knowledge base? Expand knowledge until it has a way to solve a problem...
Also let llm train with knowledge base so it doesnt have to memorize.


In mathematics its often the 'blabla A is special case of blabla B'.  Maybe goal of llm is to find the most abstract rule that still fits the data.

Skipping of the memorization phase for grokking? I mean that would be ideal I suppose as we never need the memorization

#
Maybe solution is to only allow very compressed solutions. but then, for each solution there is, make the path from one solution to the next learnable or high dimensional such that optimization can reach it. the path must be simple kinda. Seperation of concerns between what the solution space is and how it can be learned.


#
Maybe to a bit of cheating and train a model on many arc tasks untill grokking, then check how the geometry of the emebedddings or the weights look like...


#
Make the LLM knowledge wiki, but model as a graph where the edges can be arbitrary textual descriptions of the relationship between the two pages. 
Maybe then use graph theory to make the graph nicely, ie some highway connection nodes, some expections to that with direct links etc

#
Make a prompt, let llm implement, then once its finished let it write a prompt for what it just did. Then compare the two for discrapancies, use that as rl signal or just to improve or evaluate...

#
Think of neural networks in terms of being able to have a limited number of states, modeled by directions and maybe magnitude in weight space
-> model this explicitly, mlp can not output a arbitray number but only a linear combination of a set of directions

#
Regulariazation can be done through
* weight decay explicitly
* stochasticity of SGD implizitly
* Data augmentatiom
Are they all equivalent in their result? Can they be seen through a common lens? can we learn somethign from that, ie do data augmentation in a hidden layer? (Maybe for a generalizing solution, one can rotate hidden activations more or smth and still get similar results)

#
GD learning is data dependet, wd is data independent. However, can we create data such that the effect is the same as wd? would that buy us anything?

#
On continual learning and ARC - ideally we would like a model that when trained on a new task, can do that very fast. So the low rank solution should be one that can learn very fast and not one that is a specific solution to a specific problem.
Maybe train a bunch of good solutions for arc tasks and then a new model to make this good solutions fast from only the input...

#
In the artisotilan view of the platonic representation hypothesis, they postulate that in high dim spaces even random points tend to have a positive correlation. Why is that the case? is that maybe a feature of larger models?

#
Why is the brain so large, when it can only do a tiny fraction of what modern llms can, like learning infinite languages or knowing all texts...
maybe a lot more computation has to be applied to some stuff that is previously underlooked

# 
As each information is just the same in some sense, you can learn any concept or at least any concept up the tree of abstractions from any paper/information.

# 
We kinda have memorization in llms as their internal knowledge and we have reasoning in terms of their reasoning in the context window.
We kinda want both but often the memorization kicks in too much - thus we get blunt responses that just iterate the most common stuf you would say to a topic.

However, would it be possible to kind of interpolate between those two modes? such that for problems that require reasoing we would kinda turn of the memorization a bit more? 
Maybe we can do that by interpolation of the att layers vs the feed forward layers. Ie making the feed forward layers a bit less strong.

#
Trained nn have a rich structure. Can we train models like RL models inside this rich structure that can explore them and discover useful insights?

#
code als graph. jedes file/function/etc ist sein eigenes llm und kommuniziert mit allen anderen

#
Guiding copilot via md files. ie write in a file "first read this, when you implemented, read other file etc" so each file has a pointer to another file. then we can chain planing, implementing, review, simplifiy etc etc.

# Pseudo-commits
we want the model to do small commits, but not actually commit them. and really need to argue why this change is necessary.

#
Lower layers, ie in cnns, seem to 'learn the right thing' ie generalize well. maybe its because for the lower structures there is just a lot more training data, ie in a dataset that are many more edges than there are noses. Can we test that in a little experiment? ie artificially generate datasets or only train on a few lower level features.
But idk, it seems like for language we would never ever want to have "low level features". Like for vision its necessary to get into a space that has meaning, but language already is a space with meaning. So vision models and language models should be trained fundamentally different. 
In vision models we do want to also use 'low-order' information, in language we would never want that. 

# 
What even are vectors? Numbers that are associated together. But the assiciation is arbitray? could also represent individually? But what about seeing them as directions? And matrixes are just vectors that have the additional component of how the vectors is represented? -> ie a matrix can be represented by a flattened vector and and some additional numbers that indicate how the numbers should be constructed spacially?

# 
Breakign everyting down in basic differences, or in bits basically seems to make a lot of things a lot easier.

# 
Optimal coding is just to go, and ask questions, and let questions be asked and iterate..?

#
Do MI on random networks... To see if they also have the geometries and also mhm or more nicely the idea that a random big model already has all necessary algos, they just need to be trimmed.

#
Difference between a 64 bit vector and two 32 bit floats: Both have the same number of bits and thus possible states. For the 2 floats however, most bits code for magnitude, only two for direction.
Maybe llm thus encode a lot in the magnitue? so in superposition is not near orthogoal vectors, but also encoded in magnitude...?

#
Can we make the superposition more principled? by pre determininging clear clusters and then the model can assign a meaning to each cluster basically.

#
Have a look at simple algorithms like counting and how they can be represented if we keep the "llm have many features" view.

#
We can see a nn as mapping the whole space. (useful picture). In 2d basically there are regions in space that are covered by the training data. Then there are regions in space where we dont have training data, but the output is still clear. And then there are empty places in space, where we have wrong outputs and thus its like a fog of war.

#
Can LLMs reason in more than 2 dimensions? Or more easily than humans? Maybe the 2d bias helps us with certain tasks

#
Did ppl try pretraining only in RL? Probaly, can we do that for ARCAGI

#
Idea that res stream vector is the multi dim vector and that MLP is the interpreter, like subject object, or MLP is the "view"

#
What about we do a training (like in the paper where they discovery many interpretations for MI), where we train a model and then train the next model to also solve the task, but be representationally as far a way as possible from the next model and so on. Will we eventually hit a model that solves it perfectly?

-> or take an already trained model and discover all the sheafs: https://arxiv.org/pdf/2605.12671

#
Is shortcut learning always maybe that the solution actually requires multi-step reasoning, but the model with GD finds solutions that are only one or a few steps?
Design a architecture, where the model finds multi step more easily. 

#
Look more into causal methods https://grok.com/c/e41945a4-292c-4d78-9bb4-9a352cc171ba?rid=63e90617-d7ef-4bba-a9b1-a3470ad96a4e

#
Train a llm on ARC, but make the activation space really small, like 4 binary or smth, so it still learns. Then this network can be studied a lot easier...

#
Ising model, how about we model a MLP like that and then during inference just use J
Generally, maybe assume better distributions and then just use part of it

#
General theme: Instead of assuming a distrubtion, fitting it to data and then taking the distribution we can
Assume a distribution, train it on data, but then only take some part of the distribution for prediction.

#
Overview of all generalizations/fixes of correlation. Normal nn does correlation? so what are improvements

#
General theme from the ising thing: We have a observation (correlation) but what we actually care about is only a teilmenge of the correlations, so we need to find a way to disentangle

#
Decision Tree learning is also greedy and does not guarantee the simplest tree. Seems closely related to NN learning. But maybe with decision trees we can learn smth that can then translate to NN learning.
Also, what is the equivalent of the transformer architekutre for decision trees?

#
When we have a lot of training examples, it seems that we can converge to the correct solution with wd. But with little amount of data we can not, we dont find the solution that has the lowest norm. So with more data we dont get stuck in a local minimum and instead can converge to lowest norm. 
Can we thus instead create artificial training points and create them in such a manner such that the model will converge to the lowest norm? So optimizing for lowest norm kinda. Or optimize inits...

# 
Can we train both generalizing and non-generlaizing model and then test approaches to elicit the generalizing model from the non-generalizing one?

#
Automated research, but restrict the researcher to a very limited set of actions

#
StepByStep trains for one specific algo. Can we then train a second model that also solves, but in a different way than the first one?

#
When we train on one (or a few) examples, we will get some generlaizing behavior and some that is spurious. Can we then use the 2nd example to test and everything that does not work out, we delete from the model and try something new? 

#
To seperate the algorithmic step from what alagorithmic step to take, have k layers but do n forward passes, at each step a model decides which next step is chosen.

#
Can we learn a model step by step in the sense that we start with a very small model, heavily quantized etc and let it learn until convergence. here we would expect train-test loss to be the same. Then, we can start with the representations of this first model and train another small model. Etc, each one brings us closer to the real data.

#
Can we take train-test divergence as a extra loss term?

#
"To understand the transition, we look at the Helmholtz Free Energy equation, which every thermodynamic system tries to minimize:$$F = E - TS$$"... "The magic happens at one precise mathematical boundary: the Critical Temperature ($T_c$). Here, Energy and Entropy are perfectly balanced.At this exact tipping point, the system experiences a phase transition. It becomes hyper-sensitive, exhibiting spectacular properties:"
-> Very much like in meditation, "the middle way"...

#
Train a model on a idk math benchmark, once normally and once with thinking only in image tokens.

#
Read https://en.wikipedia.org/wiki/The_Book_of_Why The book of why

#
So in echo state networks we can use computation from a random network, and then only training a linear layer is needed.
Can we instead capture compute from arbitrary physical processes?
Like can we have lava lamps and use them as our black box?
Can we record stock market movement and use that?...
(thinking how, according to wolfram, the universe is a computer)