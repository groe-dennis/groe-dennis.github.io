# Computational irreducibility

for many complex processes defined by simple rules, there is no shortcut to determine the outcome other than to simulate the entire computation step-by-step

-> Normally in Science, you can reduce. i.e. calculate motion of planets with simple math, instead of having to map every atom. However, if the system is too complex this fails.

Because the observer that is trying to calculate is computationally not more powerful than the system he analyses.

Such systems can be cellular automata, that can exhibit behavior that is as sophisticated as the computations in our brains

## Implications
Limits of Prediction: It fundamentally restricts our ability to perfectly forecast complex, long-term phenomena (e.g., climate change, economic markets, or fluid dynamics) without running a complete simulation.

On Free Will: For centuries, philosophy faced a paradox: if the universe follows strict laws of physics, every action is predetermined, meaning free will must be an illusion. Computational irreducibility solves this by introducing a new kind of unpredictability.
No system outside of yourself can simulate you, because you are irreducible. Thus no one from outside can precit you and thus you are the neccessary shortcut for your own behavior (your physical brain is the fastest possible calculator of your next thought)
+ some more nuances ideas, idk here but in theory interesting

Introduces trade-off in AI between chosing interpretable but boring algo and black box but interesting algo.

# https://writings.stephenwolfram.com/2024/10/on-the-nature-of-time/
In computational termns, natural for us to think of successive states of the world as being computed one from the last by the progressive application of some computational rule. And this suggests that we can identify the progress of time with the “progressive doing of computation by the universe”.

With computational irreducability, it is impossible to skip ahead in time. 

"It’s fundamentally because of the way we are as observers. If the underlying system is computationally irreducible, then to work out its future behavior requires an irreducible amount of computational work. But it’s a core feature of observers like us that we are computationally bounded. So we can’t do all that irreducible computational work to “know the whole future”—and instead we’re effectively stuck just doing computation alongside the system itself, never able to substantially “jump ahead”, and only able to see the future “progressively unfold”."

"In essence, therefore, we experience time because of the interplay between our computational boundedness as observers, and the computational irreducibility of underlying processes in the universe. If we were not computationally bounded, we could “perceive the whole of the future in one gulp” and we wouldn’t need a notion of time at all. And if there wasn’t underlying computational irreducibility there wouldn’t be the kind of “progressive revealing of the future” that we associate with our experience of time."

# Second law of thermodynamics https://writings.stephenwolfram.com/2023/02/computational-foundations-for-the-second-law-of-thermodynamics/
 result of the interplay between underlying computational irreducibility and our computational boundedness

# https://writings.stephenwolfram.com/2020/04/finally-we-may-have-a-path-to-the-fundamental-theory-of-physics-and-its-beautiful/

Hypergraph generalizes normal graphs by edges being able to connect multiple nodes, often drawn as Venn Diagrams.

(Hyper-)Graphs are basically collections like {{1, 2}, {2, 3}, {3, 4}, {2, 4}}. Orderinging in the collection does not matter, yet the ordering inside a relation does.

Then, he applies simple rules like 
![](https://content.wolfram.com/sites/43/2026/01/sw01132026howimg2.png)

With that, from very simple graphs, very complex "organic" structures can emerge!


He says space without matter is a graph of 10^400 nodes, or even many more

"And the big question is: if we were to run rules like these long enough, would they end up making something that reproduces our physical universe? Or, put another way, out in this computational universe of simple rules, can we find our physical universe?"

Some rules make finer and finer meshes. He thinks that this leads in the limit to continous structures of the kind we are used to seeing.
-> He thinks space is ultimatly concrete

## Dimensionality of a Hypergraph
* “volume” of the d-dimensional analog of a sphere is a constant multiplied by rd.

* In a hypergraph, start at a node. Then follow r hyperedges in all possible ways. You’ve effectively made the analog of a “spherical ball” in the hypergraph. And if you now count the number of points reached by going “graph distance r” (i.e. by following r connections in the graph) you’ll find in these two cases that they indeed grow like r2 and r3.

(I guess that depends on the inital node? maybe averaged over all of them...)

* Measures this for a rule that produces a "plane" and here d goes to 2 as expected.

* Fractals have d=1.58 (the usual fractal dimension  for a Sierpiński structure)
## "Making" also matter and not just space
"But in our models there’s in a sense nothing but space—and in a sense everything in the universe must be “made of space”. Or, put another way, it’s the exact same hypergraph that’s giving us the structure of space, and everything that exists in space."

-> So what this means is that, for example, a particle like an electron or a photon must correspond to some local feature of the hypergraph

## Curvature on a hypergraph
??
But curvature gives better intuition for geodesics

## geodesics
A geodesic is the shortest distance between two points. In ordinary flat space, geodesics are just lines. But when there’s curvature, the geodesics are curved

* Geodesics are interesting, because it's the path that light travels. And if there is a curvature in spacetime, light will also move in the curved dimension

* With his theory, he can reproduce the Einstein equations

## On Calculus
Calculus does not work on the inf limit hypergraphes as it does on continous spaces. Instead a generalization of calculus needs to be found. He argues that that is Geometric group theory. 

## Time
* Time is just the progressive application of rules

* But there is a subtlety in exactly how this works that might at first seem like a detail, but that actually turns out to be huge, and in fact turns out to be the key to both relativity and quantum mechanics.

* Rules are defined over all nodes, which nodes to pick is arbitrary. Thus, a tree of possiblites emerges

-> There are many paths of time, paths can merge

* To an observer embedded in the system, there is still just a single thread of time

More: Time is about causal relationships and even if different paths are followed, the causal relationship can still be the same.

## Causality
For rules such as {A → BBB, BB → A} it happens that there are at first two possibilites, but then the branches converge again
-> He calls that "causal invariance" (elemental for relativity and quantum mechanics)

* New construct: The causal graph: 
    * Nodes: Update events (a edge in the original Graph of rule application)
    * Edges: The causal dependency between the events, which event needs to occur first before another event can occur 

In rule rule BA→AB (sorting) causal invariance becomes apparant, because no matter what we path we take, we always end up with a sorted list

Same causal invariance in math, like  (*x* + (1 + *x*)2)(*x* + 2)2** You could expand one of the powers first, then multiply things out. Or you could multiply the terms first. It doesn’t matter what order you do the steps in; you’ll always get the same canonical form.

* When one thinks about parallel or asynchronous algorithms, it’s important if one has causal invariance. Because it means one can do things in any order—say, depth-first, breadth-first, or whatever—and one will always get the same answer.

## Connection to Einsteins relativity
Very interesting chapter "Deriving Special Relativity".

Basically, he argues that we slice a causal graph as observeres, and call it a "step in time", with multiple things happening in each slice. 
(the complete causal graph is timeless, and time is a concept to understand it better...)

The slicing however has to be done in such a way that it does not break causality (thus its horizontal)

Then he imagines that a observer moves horizontally. Then he has to shift the slicing (didnt't completly get it from here on)

## What Is Energy? What Is Mass?

"A spacelike direction is one that involves just moving in space—and it’s a direction where one can always reverse and go back. A timelike direction is one that involves also progressing through time—where one can’t go back."

... more stuff I did not understand

## Rulian Space

* Supposes that our universe can be described by a particular rule. Then questions how to find it or if its even findable etc

* Then proposes what if there isn't one single rule, but instead all conceivable rules are used
-> imagine making a multiway graph of absolutely everything that can happen—including all events for all possible rules
(will oc have causal invariance bc rules and counterrules)

-> calls this rulian space

for observers in rulian space:

"It’s a strange but rather appealing picture. The universe is effectively using all possible rules. But as entities embedded in the universe, we’re picking a particular foliation (or sequence of reference frames) to make sense of what’s happening. And that choice of foliation corresponds to a description language which gives us our particular way of describing the universe."

-> the universe is a universal computer

"I’ve always assumed that any entity that exists in our universe must at least “experience the same physics as us”. But now I realize that this isn’t true. There’s actually an almost infinite diversity of different ways to describe and experience our universe, or in effect an almost infinite diversity of different “planes of existence” for entities in the universe—corresponding to different possible reference frames in rulial space, all ultimately connected by universal computation and rule-space relativity."

## The Challenge of Language Design for the Universe
"What does it mean to make a model for the universe? If we just want to know what the universe does, well, then we have the universe, and we can just watch what it does. But when we talk about making a model, what we really mean is that we want to have a representation of the universe that somehow connects it to what we humans can understand"


"And I now view the effort to find a fundamental theory of physics as in many ways just another challenge in language design—perhaps even the ultimate such challenge."

"In designing a computational language what one is really trying to do is to create a bridge between two domains: the abstract world of what is possible to do computationally, and the “mental” world of what people understand and are interested in doing."

## Outlook

There are hints of string theory, holographic principles, causal set theory, loop quantum gravity, twistor theory, and much more. And not only that, there are also modern mathematical ideas—geometric group theory, higher-order category theory, non-commutative geometry, geometric complexity theory, etc.—that seem so well aligned that one might almost think they must have been built to inform the analysis of our models.


# https://writings.stephenwolfram.com/2025/05/what-if-we-had-bigger-brains-imagining-minds-beyond-ours/

* Imagines what could be possible if a brain had a lot more neurons

* Generally, there is computational irreduceability. However, in every computational irreducable system "there must inevitably be an infinite number of “pockets of computational reducibility”, in effect associated with “simplifying features” of the behavior of the system."
-> The more pockets a brain can hold in mind, the mroe sophisticated it is

* Abstraction (i.e. from lion, tiger to "big cats") is useful if it’s possible to make collective statements about those general things (“all big cats have…”), in effect providing a consistent “higher-level” way of thinking about things.

* He says language has 30.000 words (or concepts).LLMs in their weights discover many more

* Most obvious use of language is to transfer thoughts from one mind to another (brains have many concepts, languge is the compressed transfer with a common interface)

* Wrangles with if bigger brains would be incomprehensible enough, or, if our current concepts are universal enought to be able to understand it meaningfully

## How brains seems to work
The total number of neurons scales roughly with the number of “input sensors” (or, in a first approximation, the surface area of the animal—i.e. volume2/3—which determines the number of touch sensors).

* animals can’t “go in more than one direction at once” that brains seem to have the fundamental feature of generating a single stream of decisions. And, yes, this is probably why we have a single thread of “conscious experience”, rather than a whole collection of experiences associated with the activities of all our neurons. And no doubt it’s also what we leverage in the construction of language—and in communicating through a one-dimensional sequence of tokens.

* Bigger and more developed brains typically seem to support larger amounts of working memory. Adult humans can remember perhaps 5 or 7 “chunks” of data in working memory; for young children, and other animals, it’s less

* As we try to reflect on what our brains do, we’re most aware of our stream of conscious thought. But that represents just a tiny fraction of all our neural activity. Most of the activity is much less like “thought” and much more like typical processes in nature, with lots of elements seemingly “doing their own thing”. We might think of this as an “ocean of unconscious neural activity”, from which a “thread of consensus thought” is derived.

## Language and Beyond

* Central aspect of brains is that they group stuff together / ignore unneccessary details
-> equivalences many different inputs together (From Observer Theory)

* With deeper brains he images that they would have more concepts, allowing for more efficient communcation, like technical jargon

* Or with more working memory more deeply nested phrases

* Brains seem to be of the kind "single thread of decisions from sensory input". This is opposed to systems generally seen in nature, that don't have the single stream constraint. 

* Principle of Computational Equivalence says it’s easy to have a system that can deal with arbitrary computation. It’s just that such a system is not “brain like” in its behavior; it’s more like a typical system we see in nature.

* Another generlaization could be to make the stream parallel instead, effectively enabling a "graph of thinking"

* He imagines communcation via images and not the single stream. I don't quite understand, but maybe bc images are parallel pixels. He says gestures and body language could be a way to communicate in pixels

## Sensors and Actuators

* We compress a lot of of the sensor data, based on what we needed for survial

* In our digital era though we could perhaps use a lot more features

* We could even transform stuff like fluid motion into pictures and then develop features for exactly that. Maybe here concepts like subject and object would even not be neccessary anymore...

* Also we could connect different actuators (For example, while humans can control helicopters, they never managed to control quadcopters—at least not until digital flight controllers could do most of the work. )

* Animal can only sense stuff where they are. One could imagine however a distributed, like a IoT, network so you get brain input from all over the world/universe

* Or chaning the timescale of input (at a million times faster, we could detect individual photons, we might be able to detect quantum stuff)

## Abstraction

* The more abstractly we try to think, the harder it seems to get. But would it get easier if we had bigger brains? And might there perhaps be fundamentally higher levels of abstraction that we could reach—but only if we had bigger brains.

* One abstraction occurs when we notice a set of things being similar, then they can be "packaged up" in a word
-> Abstractions/concepts/ideas can be self-reinforcing, such as with the concept blog. As soon as the concept exists its easier to think about and we tend to see it more and more in the world

-> But what really makes abstraction take off is when we start building a whole tower of it, with one abstract concept recursively being based on others.

* Abstractions define a concept independent of underlying objects, such as measurements of weight independent on what is weighted (figs or wool etc)

* Some concepts you can only explain by going up the tower of abstraction. Some seemingly abstract concepts can however be explained well from "raw experiential concepts"
-> However, when explaining abstract things, one can do that by analogy to more familiar, concrete things

* One can see abstract concepts as a reflection of pockets of reducablity, "jumping ahead" without having to explain everything from the ground up

* ruliad can be seen as the  entangled limit of all possible computations, but it can also be seen as the limit of all possible abstractions
-> he argues that it’s because brains are the way they are that we perceive the laws of physics that we perceive.

* For bigger brains, if they are not fundamently different, the same thing would apply. Because for the core laws of physics, only two properties of the observer are important: First, that they’re computationally bounded. And second, that they believe they are persistent in time, and have a single thread of experience through time

"Would this be different with bigger brains? Perhaps a little. But anything that something brain-like can do pales in comparison to the computational irreducibility that exists in the ruliad and in the natural world. Nevertheless, with every new pocket of computational reducibility that’s reached we get some new abstraction about the world, or in effect, some new law about how the world works."

* He asks if reaching higher levels of abstraction is a matter of education, and to what extent it requires additional intrinsic capabilities of a brain. He suspects both, Sometimes it’s really just a question of knowing “where that pocket of reducibility is”. But sometimes it’s a question of navigating a network of pockets, which may only be possible when brains reach a certain level of “computational ability”.

* Our brains are incapabale of storing all the information of our civilization. Bigger brains however could do that
-> They could then make "grand analogies" from different branches of knowledge (he made one from special relativity to biology)


## Computational Language

"In some sense, human language tends to “work with data”, applying a simple analog of code to it. Our computational language can “work with code” as “raw material”."
(human language is working with concrete objects, in computers we can work with functions as objects)

In the wolfram language, everything, including objects, actions, etc. are symbolic expressions

-> There is no need to use names, everyting can be defined by its interaction bascially (combinators in math)
-> Combinators have been around for more than a century. But they are almost impenetrably difficult for most humans to understand (Somehow they involve too much “pure abstraction”, not anchored to concepts we “have a sense of” in our brains.)

When people learn it it takes time, After one groks one level of abstraction, that seems to have to “settle” for a while before one can go on to the next one.

-> He suspects having a bigger brain could speed this up, but unlikely (as as soon as one goes to far out, one ends up with a brain that is irreducable and thus can not useful (or smth like that))

"One of the core features of the Wolfram Language is that it represents everything as a symbolic expression. And, yes, symbolic expressions are formally able to represent any kind of computational structure. But beyond that, the important point is that they’re somehow set up to be a match for how brains work. (https://www.wolframscience.com/nks/p620--human-thinking/)"


* symbolic expressions can be thought of “grammatically” as consisting of nested functions that form a tree-like structure; effectively a more precise version of the typical kind of grammar that we find in human language


# Wolfram Language
It is based on symbolic expressions

In python, you mostly care about the result of a function. Even when you pass a function as a argument, you still don't know the internals of the function.

in wolfram, a function however is a symbolic expression (a tree). Other functions can manipulate the tree. This is usefull i.e. to take the derivative of f(x) = x² + 1, wolfram represents it as 
Add(
    Power(x,2),
    1
)
And this allows the differentiation function to apply rules like 
d(x²)/dx -> 2x
d(constant)/dx -> 0

* In python you could get those symbolic expression stuff by
'''
import inspect

source = inspect.getsource(square)
'''

Then one is working with the source text itself, not just the evaluated argument of the function.

So wolfram is able to rewrite the the parsed representation of the program

# Ideas
Looks very much akin to contemplative structures

* Maybe the recent RNN hype because this "time" can not be parallized, bc of computational irreduceability there needs to be this unfolding in discrete steps (which corresponds to wallclock time) that needs to unfold

* Whenever there is causal invariance we can describe the outcome in a abstract word? Such as for BA→AB we can say "sorting in alphabetical order" instead of describing the behavior of the rule by its complete tree (difference between causal invariance and computational irreducability, even though we can not sort the list without going throught the calculations (or we can skip steps for very specific list, like only A B oc because of heuristics, but not in the general case) we can still describe what the outcome will be)

* "What does it mean to make a model for the universe? If we just want to know what the universe does, well, then we have the universe, and we can just watch what it does. But when we talk about making a model, what we really mean is that we want to have a representation of the universe that somehow connects it to what we humans can understand. Given computational irreducibility, it’s not that we expect a model that will in any fundamental sense “predict in advance” the precise behavior of the universe down to every detail (like that I am writing this sentence now). But we do want to be able to point to the model—whose structure we understand—and then be able to say that this model corresponds to our universe." 
-> is next token prediction the right thing to do given this idea

* Are neural networks observers?

* If concepts are language and llms have concepts interanlly, what is the equivalent inside a llm of verbs, adjectives, nouns, grammer, tenses, subject-object etc. Like maybe we can see a llm activations as a stream of non-linear language

* In programming langues one can not only pass variables, but also pass functions. How would that work in a llm?

* Wait ok body language communication as more efficient? we have that and the interpretation is called "intuition". So basically we have the thread to thread communication of words but we also already have the "graph" communication that he eludes to via body language etc. This goes more of less directly from concept space in one brain to another, skipping the requirements of single threaded communication...
-> maybe that is also related to communicating in functions not in concepts, point above (ie a image can be represented as a function or smth)

* Another feature of larger brains? : faster learning, more adaptivity, even more than we do for a digital era

* is there a fundamental difference between object/concept and function? is a object the easiest function?

* Maybe there is a difference between defining a function as a function of functions and then executing that function? Ie actually in llm research we would like to learn the code in the sense of a subroute thread, and then at a seperate step we would execute and evaluate that function. Somehow like we seperate training and inference...
Can we make this seperation more explict in the architecture design?
Like the reasoning is this, but a bit ad hoc on top isntead of proper function chaining maybe

* If I understand correctly, Wolfram language only has functions and not objects themselves. So everything is defined as a doing, very similar to what the contemplative tradition says

* "rewriting the parsed representation of the program" Wolfram, this is bascially also what a llm is doing when coding?

* The shift between python - wolfram. I imagine nn as python-like. are there wolfram-like neural network architectures?
What can Wolfram stlye language tell us about neural network interpretability?