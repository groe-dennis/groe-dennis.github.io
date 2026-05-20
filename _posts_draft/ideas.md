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