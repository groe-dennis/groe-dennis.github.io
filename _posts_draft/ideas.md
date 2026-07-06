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

#
Okay, so computing with physical processes is a thing already, called reservoir computing. But they are limited to systems, where we can input and output smth. This is a big restrictions.
Could we instead work with just a video of a process, like a video of a wood? Where a model can, like a expert, send to a specific frame in the video, and then the output would be the next frame (or the next few frames or smth)
-> isnt this like what llms already do? like if we see text as the physical process. But could we make a llm that maps to a sentence in the training data?
-> the lookup on the tree video would be similar to just calculating another matrix... mhm can we just have lookup tables inside a llm? skiping layers entierly? I think thats also a established idea? but maybe in a looped form, like input, then lookup then output this in a loop

-> can we use random network once, and run it with every possible input? and then we can put that in a db instead of needing to run it every time. Basically we amoritze costs.

-> or like basically we take a trained model and we record its output for n inputs. Then for a new input we can kinda do it like skip connections, where we only need to learn the part that is not yet recorded.
#
Have n llm each representing one paper. then open a group chat, every time a model has smth interesting to say, it will join the conversation...
should be possible with small models actually.

# 
Learning is best if you start with what you already know and then introduce a new atom of knowledge by contrasting it with the known to see where its different

#
Maybe what I do here is quite nice in the sense that I only commit stuff that I understand. And this can be used as a LLM knowledge base. This might be a good extension of brain as I can query the llm then. What would be bad is to commit stuff I dont understand. 
Similar to how the brain works? I mean I don't rememeber everything all the time anyway, so when a prompt is given I can either go to external or internal memory, but no matter where the information then comes from, I will be able to understand it and use it skillfully.

#
What even ist regularization philosophically? Is it there to be able to compare between instances? probably also something deeper related to nn

#
Learning stuff, the highest ROI might not be knowing stuff oder understaning it, but to get the context in which the stuff came up and to get what the ableitungen to other knowledge is. To make it actionable. Maybe this can be modeled with a programmatic function, maybe its good iff it helps you do/learn other stuff.

#
RL training should have a back-questions part implemented and simulated by giving questions that are ambigious, but a oracle llm has access to them.

#
Can we reframe GD as Multi-armed-bandit?

#
Okay so we know from Matroids that if the Zwischenschritte of an algorithm are all independent, that the greedy algo of just taking the lowest loss works. GD imo can be seen as a greedy algorithm. Large vectorspaces have many vectors that are almost independent. Research shows that minima in NN are connected by simple curves where the loss stays low and flat minima are often prefered. 
Now Matroids greedy algo classically are like find a basis with minimum weight, and the greedy algo is to just pick new independent ones and always the lowest one remaining. 
If we change the objective of GD from finding the weight with minimal loss to finding the minimum over all seen weights (if we assume they are independet, finding the basis), then we have the same setup and we know that GD is optimal in as sense, and bc larger networks are more likely independent that would explain smt.
But also if we optimize all seen weights, that is more like finding a low loss basin, corresponding to good generalization?

-> now next to theory, can we make smth with that? can we make the seen weights optimize for independence or smth etc
* 

0ter Punkt
Wir nehmen an wir haben größen Lösungsraum. In dem gibt es viele perfekte Lösungen. Wenn wir aber in diesem Raum Greedy suchen, kommen wir nicht an die beste Lösung
Stattdessen schränken wir den Lösungsraum ein. In dem eingeschränktem Lösungsraum gibt es immer noch viele perfekte Lösungen. Aber in diesem Lösungsraum, (zB wegen Matroid eigenschaft) ist ein greedy algorithmus (zB gradient descent quasi) optimal.

Ein größeres Modell schränkt deswegen den Lösungsraum ein, weil jeder schritt quasi unabhängige weights findet. Also die Weight historie ist ein orthogonalsystem. 

Okay also erster Punkt: Vielleicht wollen wir nicht einfach nur die besten weights, sonder wir wollen das beste "basin" finden. D.h. eine Menge an weights, alle mit einem gradient Schritt verbunden, die zusammen den geringsten loss haben. Das würde für die Theorie sprechen mit dem flat loss regions die durch einfache kurven verbunden sind.

Zweiter Punkt von der Matroid theorie: 

Also wir haben einen großen Lösungsraum, es gibt viele Lösungen die perfekten loss haben (gehe davon aus das perfekter loss mit regularizer auch perfekt generalisiert)

Größere Modelle haben in ihrem Lösungsraum auch den Lösungsraum kleinerer Modelle, also alle funktionen die kleine modelle abbilden können, können auch größere.
Angenommen auch kleinere Modelle haben perfekt generalisierende Lösung, aber wird nicht gefunden.

Wenn wir einen greedy algorithmus jetzt auf dem 

# 
Unterschied "simpleste lösung" vs "den besten/simplesten schritt von einer ausgangslage machen" - unterschied occams razor vs wie den menschen häufig falsch verstehen

#
Annahme: Die eigentliche Intelligenz des models liegt im base model, post training erklärt dann nur noch welches verhalten man haben möchte (vlt nur teilweise so)
Aber kann man dann irgendwie die Kraft eines Base models nutzen und in die Bahnen eines Posttrained models tun? Also das Postrained model vlt quasi als einschränkung in der Antwort sehen, aber innerhalb der Einschränkung darf das base model frei entscheiden? udn dann könnte man viele basemodel miteinander verbinden oder so?

#
"Intelligence becomes most apparent under restrictions" -> Kann man das für LLM training benutzen indem man irgendwie den Lösungsraum einschränkt

#
Can we train on arc and then generate more input output examples. with those more we then train a new model and check how well it performs on the real data. use that as a signal.
bascially, we test how well the model has approximated the rule, by letting the rule be found out by a data learner. 

#
Maybe we can look at text data as a result of a large number of functions.
Like if we were to run a codebase for a while we would get a loot of logs and thus that would be like the data we see...

-> eval gives logs and makes the llm reproduce the codebase...
-> if training gets the correct rules, the codebase will be reconstructed and thus perfect generalization
(maybe some kind of reflection problem here do we just want the model to be a codebase or a model that can reconstruct codebases...)

#
Can we have a model that outputs the weights of a model (maybe compressed, but we do want it compressed anyway), like it ouptus lora weights or smth.

#
Maybe the main problem with models is that the intent is not brought about correctly. Ie when doing an image I have something specific in mind because I have a certain goal with it but the model does not have this contextual understanding. solve with back questions?

#
Let model decide on which and how much tokens to train. ideally then it would only learn useful information and discard information that will not be useful later on.
thus maybe also sample efficiency will go up

questio oc what do even train for? when do we want to learn from a token and when not? 
ie val loss is misleading? because we dont actually want the model to be able to predict every token in the val set

#
Maybe we can do this let model decide what to train in a way that we make like a test, that checks if a model has access to a certain token prediciton, does that make some other text easier to predict?
so maybe let a trained model give some text and mask some token so which tokens are needed to predict another token.
OC there might be a difference between what we want to keep and what is just needed for a specific text...
maybe which token is important to learn for some completly different text.-> ood generalization
maybe train with that first and only after do normal training (to recover normal model behavior, but due to prepretraining it will only learn good stuff)
-> generally maybe do the paramter golf but find a good way to do pre-pretraining
-> maybe here we can differentiate between memprization and reasoning, if we can construct prepretrain so that we only have reasoning, we can add some slight memorzation on top.
how usefull is some data to predict other data, that is completly unrelated is that reasoning?

# 
Physical reservoir computing with microphones and camera seems like a cool project

#
can we do smth when we have few input variables, and then we just have a massive dataable that kinda skips llm layer computations? 

#
An algorithm is independent of data. howver llms are very dependent of data. how to construct smth similar that is independent of data? a few tokens that always get uniform token as input, then other tokens can depend on them? maybe thats a better seperation? 

#
Train a nn to have the same loss for al training examples and tokens. then move down only in the path where the the loss is always the same for all token.

#
Simiallry, train a model to have same loss for different levels of quantization. Or can we train such that we fix that the model must have the same loss for different levels of quantization/compression? so we can still search in the big space but we reject samples that can not be compressed. Difference to QAT? 

#
Can we train a model that has access to a text corpus and learn to query it like a database? so the model never really has to store in weight, it can just use computation from the data. Can this be done for ARC?

#
"its not about solving a problem, its about while solving a problem to figure out smth that helps with other tasks. Like Tao said for math. or like epipliexity"
How about we construct a game. one model generetes arc data, the other trains on it. so they have this continous game going, and the only thing they get as input is the complete loss of all the train examples. (but diff to just rl/es with complete loss?)

# 
It seems like humans always have a goal in mind, so a loss function/reward function. And then optimize for it until they are satsifed. can we do the same with a model? when it first gets a prompt it constructs its own reward function, then optimizes for it. Then it compares to ground truth. That way, with data we optimize not modle behavior but the models ability to construct reward function

#
What if we dont train a model to have high loss and instead we train a model to make a function that has a arbitrary loss value? this is a much harder task I assume, especially for combinations where the loss value needs to be the same for all tokens. Input is then loss value(for each token)+normal input, output is diff to loss value.

#
Its not neccessarly that for generalization we need a model that is compressed, but we need a model that is compressible... So the calculations it does might still be complicated? yet its simple idk

#
Do Literature search on the core differences between memorization and reasoning and then train for that
-> M

#
What I done before, training with arc data but also with a lot of other data to force the llm to find a simple solution to the data... just with more training time? maybe train for longer on the examples so nn really has to memorize...
also maybe first train on arc data to get loss 0, then just train such that loss also stays at 0
-> more interestingly, are there scaling laws when we add new random data?

#
https://gemini.google.com/app/cfd40359cffedd84?hl=de when compressing a model, memorized facts deteriote fast, while reasoning is more compressible.
-> can we do a curriculum of first training, then compressing, then training again etc etc?

#
We can actually see a model as a addition of matricies: A + B + C or maybe also as a matrix product A * B * C, because this is what training does basically.
1. Can this be done with input data? can we first do input like A * Input such that we then get the W with which we do normal training? maybe transformer does that already or fast weight adjacent. but that is kinda like "choosing the algo" vs "computation of the algo" maybe seperating that would be good.
2. Can we train with A + B so we train all those weights, but then during inference we only use the combination? Like lora but expanding parameters instead of reducing them. so we have more paramters to train basically, idea that that way a simpler slution can be found but after we can easily compress. Does this even make sense?
#
Can we start with a highly compressed network, ie we only allow for 100 bits but then we complexify through lookups and loops etc?
(each matmul is kinda a lookup and then linear combination?...)