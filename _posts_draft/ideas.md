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