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
