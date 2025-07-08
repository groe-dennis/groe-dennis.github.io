---
layout: post
title: Large Bounce
subtitle: On expansion and contraction in neural network training
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/path.jpg
tags: [draft]
---

------------------------------------------DRAFT------------------------------------------

# On expansion and contraction in neural network training

# Purpose

Explore if ideas of expansion and contraction lead to better neural network model training. 

# Inspiration

Big Bounce universe model
Expansion and contraction in meditation
Workflow in engineering: start of building something that works, might be quite complicated and not clear what each part does. But then, there is a working prototype and from that, one can reduce and ablate parts to find the core that actually works. Then the cycle continues. Famous example is the rocket design of the SpaceX Falcon.

# How this might translate to neural networks

Double descent phenomena: First the model finds corse solution to the trainset by fitting all the data. But then the capacity does not work out anymore and based on this first solution the model finds simpler solutions that also generalize better.

Idea: To train a model on areas without infinte data, take the trainset and first train the model. Then, prune/simplify it to the point where the train accuracy again is in line with the test accuracy, so no overfitting. Then add complexity again by expanding the model or by training again. Repeat. (There is previous literature on this)
