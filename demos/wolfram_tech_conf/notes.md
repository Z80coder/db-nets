# Neural network research with the Wolfram Language

Hi everyone, my name is Ian Wright, and I'm a Machine Learning Engineer at GitHub.

I want to talk about my experience using the Wolfram Language for neural network research.

# Abstract

In particular, I want to relate how I used various language features to rapidly explore and prototype a new kind of neural network, which can learn highly compact, non-differentiable Boolean functions.

So let's get straight into it.

# R&D goals

A neural network is a function with learnable parameters, called weights. Training iteratively adjusts the weights by backpropagating an error signal to minimize a measure of loss on the training data. Essentially, this process performs gradient descent in a high-dimensional weight space.

Neural networks must therefore be differentiable functions.

But this creates some problems.

We can't learn discrete, non-differentiable functions, such as boolean functions. Discrete functions give better bias for some learning problems.

Also, nets can be huge because we must store floating point numbers for every weight. And they can be expensive to query because we must perform large numbers of floating point operations.

There's a lot of work to reduce net sizes, such as weight quantization, which represents floats with less bits. But quantization is a kind of lossy compression which degrades the network's predictive performance.

I wanted to see whether we could have our cake and eat it.

I wanted to see whether we could define a new kind of neural network, which is differentiable, so we can train it efficiently with backpropoagation, but post-training convert it to an equivalent boolean function with 1-bit weights without any loss of performance.

This is the idea for db-nets.

A db-net has two aspects: a soft-net, which is differentiable, and a hard-net, which is a boolean function. We train the soft-net as normal, convert its weights to 1-bit booleans, and bind them with the boolean function. And this boolean function behaves in exactly the same way as the soft net.

So this is what I set out to do. I wasn't sure if it was possible. And I knew I had to rapidly explore, and test, lots and lots of ideas.

So that's why I chose to prototype in the Wolfram language.

# Hard and soft-bits

To get started, we need to define two kinds of bits.

A soft-bit is a real number between 0 and 1. A hard-bit is a boolean value.

A soft-bit convers to True if its greater than 1/2. Otherwise it converts to False.

This "Hard" function performs the conversion.

We want the soft-net and hard-net to be semantically equivalent. 

The LHS of this equation says: we supply an input x to the soft-net and get an output. We then harden that output to boolean values.

The RHS says: we harden the same input x to boolean values and then supply this to the hard-net to get another output.

If these two outputs are identical for all possible inputs then the soft and hard-nets are semantically equivalent.

This is a hard-equivalence requirement that db-nets need to satisfy. The soft-net needs to be differentiable but nonetheless harden, without error, to a boolean function with 1-bit weights.

And this requirement turned out to be surprisingly hard to satsify! I had to try lots and lots of different approaches.

Over a period of a few months I prototyped -- no exaggeration -- 100s of ideas.

The Wolfram Language, because its a single, unified system that provides everything immedaitely out-of-the-box, and it all works together, I essentially had a playground to try all kinds of wacky and wonderful ideas. The ability to quickly dynamically visualise functions, and their gradients, was really helpful, providing quick intuition. Most ideas didn't work. But I could prototype really fast, and discard them.

# Learning to AND

Here I'll briefly explain my eventual solution.

Let's start with a simple problem that will help illustrate the general solution.

Say we want a network component that can learn whether to mask an input value, x. If a learnable weight, w, is True, then the input value is passed through. But if the learnable weight, w, is False, then the input value is masked to False.

So we want to learn a boolean weight, w, for the boolean function And. And obviously logical And is not a differentiable function.

# Product logic

Real-valued relaxations of boolean logic already exist. For example, product logic implements logical And as multiplication.

In this plot the horizontal axis is the soft-bit input x. The vertical axis the output.

Here the weight is 1 or True. And we can see that the input-value passes-through to the output, perfectly. When the input is low the output is low, and when its high the output is high.

But look what happens when we decrease the learnable weight. It's about 0.6 now. Which means the weight is hard-equivalent to True. But if the input is, say, 0.6, then the output is 0.4, which is hard-equivalent to False. So ProductAnd, in this situation, is not hard-equivalent to logical And.

That's a problem.

We can visualise this problem all-at-once, using Wolfram's countour plot.

Again, the horizontal axis is the soft-bit input. But now the vertical axis is the soft-bit weight. The countours represent the output of ProductAnd. And I've coloured all outputs that harden to True as pink. And outputs that harden to False as Gray.

And here you can see the problem even more clearly. For hard-equivalence to And the whole top-right quadrant must be entirely pink. The hard boundaries at 0.5, where a soft-bit chantges from True to False, need to be respected. But they're not.

Now, the countour lines are curved, which means that ProductAnd provides a gradient everywhere, in other words is differentiable in the right way.

But it's not hard-equivalent. So it won't do. If we used it at training-time, and then hardened the learned weights, we'd get different behaviour.

# Godel logic

There's another relaxation of boolean logic named after Godel. It defines a differentiable version of And using Min.

When the weight is 1 it all looks good again. What happens when we decrease it?

Very different behaviour! Unlike before it respects hard-equivalence. In fact as the weight just gets below the hard-boundary at 0.5, the output becomes less than 0.5. And so it masks the soft-bit input.

We can see this more clearly again using a countour plot. This time the top-right quadrant is all pink, which means Godel-and is hard-equivalent to logical And.

But there's a problem! The countour lines are straight not curves. That's because the Min operator essentially selects either the input value, or the weight, as its output. So any change in the unselected value doesn't change the value of the output. And so there's no gradient! There's no gradient information to backpropagate in these circumstances. And that's bad for learning.

Godel-And is hard-equivalent, but it isn't gradient rich.

So this won't work!

# Margin packing

For db-nets we want real-valued relaxations of logical operators that are both hard-equivalent and gradient rich.

We saw that Godel-And is hard-equivalent to logical And. So let's use Godel-And to define a "representative bit", which is guaranteed to be hard-equivalent to logical And.

Here you can see the representative bit as a bold vertical line. The two manipulable parameters are the input value and the weight.

If we set the weight to 1 this function should act as a pass-through. And you can see that it does as I vary the input x.

But notice that there's always a margin between the 
representative bit and the threshold at 1/2. Any output-value between the representative bit and the threshold at 1/2 will still be hard-equivalent to logical And.

So we pack that margin with extra, gradient-rich information. In other words, we compute an augmented-bit that always varies with both the input and the weight values.

So even when I vary a parameter ignored by the Min operator the augmented bit still varies. Yet it preserves hard-equivalence. The mathematical details of margin packing aren't important now.

But in this case we get a new, piecewise function.

Which we can plot as a countour plot. So this new function is hard-equivalent to logical And (because the top-right quadrant only is entirely pink). But notice that the countour lines are curves -- which means there are gradients everywhere! 

(Actually this function is differentiable almost everywhere, which is sufficient for backpropagation to work.)

We construct db-nets by composing margin-packed, differentiable analogues of boolean operators into multi-input neurons, then into multi-input layers, and then finally into complete net architectures.

# Neural network layers

So we're off. 

I implemented a db-nets library that uses the Wolfram Language support for neural networks to create new kinds of differentiable boolean logic layers.

We can have learn-to-AND, OR, NOT layers, and all kinds of more exotic variants.

Here I want to look at a differentiable Majority layer, because it shows just how far we can push Wolfram Language's support for neural networks, and get it to some rather surprising things.

The boolean Majority function outputs True if the majority of its inputs are True. It's a discrete analogue of a threshold function, and very useful for learning problems.

Majority can be computed in terms of AND and ORs. But the number of terms grows exponentially with the size of the inputs. And no algorithm exists for finding the minimal boolean representation of Majority. In neural net applications we may want thousands of inputs bits to a single majority neuron. This rapidly gets explosive to compute at training time.

But notice that if we sort the soft-bits in ascending order then the "middle bit" is representative and therefore hard-equivalent to boolean majority. 

And, remarkably, we can easily define a network layer using Wolfram's FunctionLayer and PartLayer. The FunctionLayer sorts the inputs, and the PartLayer picks out the "middle bit". And this is 2 lines of Wolfram code! And the Wolfram Language takes care of everything, including compiling it down to differentiable code. The error is then backpropped through the representative bit.

But we can margin-pack this bit to backprop error through all input bits.

So we avoid the exponential blow-up by paying the run-time cost of Sorting.

## Soft net

Let's define a majority layer that takes 8 inputs. Every time you define a db-net component you get 2 things: the soft-net and the corresponding hard-net.

Let's look at the soft-net. Wolfram's out-of-the-box net visualisations really help when wiring things together. And it's pretty incredible how Wolfram Language just wires this all together into a differentiable layer.

And we can interactively test our new layers, to see how they behave.

Let's define some test input.

And then let's see what the layer outputs. 

## Hard net

It's just a Wolfram function (with some extra complexity to allow it to be composed into a hard network). But all it reduces to is the boolean Majority function.

That means, post-training, when throw away the Sorting, because it's all hard-equivalent to boolean Majority, which is extremely fast to compute (it's just counting bits).

And we can check that it behaves the same as the soft-net. 

It's pretty incredible that we can define a differentiable analogue of boolean majority.

The db-nets library uses other advanced features, such as Wolfram's CompiledLayer to define special neurons that behave differently in their forward and backward passes. This kinds of control is needed to ensure db-nets backpropagate the right kind of error signal. The details aren't important. But what is important is that Wolfram Language allows you very deep control, which is essential when exploring novel research ideas.

# A classification problem

OK, let's put this all together, and quickly look at a toy-example of db-nets working in practice.

Let's get some data using the Wolfram data repository.

Basically it's table describing features of cars. And the aim is to predict whether a car is worth buying, that is one of
- unacceptable
- acceptable
- good
- or very good.

So it's a small multi-class classification problem. With numeric and categorical features.

# A classification problem

We'll use a function from the Wolfram function repository to split the data into a train and test set.

And we'll use NetEncoder functionality to automatically convert the input features into an indicator vector of booleans.

And then we compose the encoders into a single input layer that outputs a single vector of booleans. It's really cool how easy it is to do this, and how we can clearly visualize what's going on.

# Define db-net

We'll now define a complete db-net architecture to learn this classification problem.

[Skip forward to start training and then skip back to here]

It has two layers, an logical OR layer, followed by a logical NOT layer. Each layer has 64 neurons.

We have 4 output ports for each class. Each output port is a vector of bits. Essentially we add up the bits in each port, and interpret them as relative class probabilities. Then we use a standard cross-entropy loss for training.

# Train db-net

Training on my laptop's GPU just works out-of-the-box.

And once trained, we can extract the trained net.

# Evaluate db-net

Let's do a quick evaluation on the test data.

Here I use the ClassifierMeasurements function to evaluate the trained soft-net and a hardened version of the soft-net. The hardened version is exactly the same as the soft-net except we harden all its weights to 0s and 1s. But, at this stage, the 0s and 1s are still represented as floating-point values.

The confusion matrix shows that the db-net has solved this problem, and that -- even with hard-weights, it behaves exactly the same.

# Bind weights with hard-net

But what we really want is to extract the boolean function that predicts whether a car is acceptable or not. We extract the boolean classifier by binding the trained weights from the soft-net with the corresponding hard-net.

What we get is a Wolfram function with associated boolean weights. This is the hard-net representation.

And we can actually symbolically evaluate it, to see precisely what boolean function has been learned. Here the b's correspond to the 21 inputs bits, which represents featrures of the car, and the logical ORs and NOTs represent the boolean functions computed on the net's output ports. Essentially the net has learned to combine different features as evidence for different classification labels.

# Evaluate boolean classifier

But is this boolean function really hard-equivalent to the soft-net? Well we can use it as a classifier, and therefore evaluate its performance on the test data.

It gets an accuracy of 99.4%, which as you can see, is identical to the soft-net's performance. So there is no loss of accuracy when using 1-bit weights.

And we get this accuracy with a much, much smaller net at query-time. In fact the weights for this classifier only consume 0.2 k.

A minimal-size MLP on this same problem consumes about 16 k.

The db-net has comparable performance but is much more compact. Also boolean and integer operations can be cheaper than floating-point operations. So this makes db-nets a potentially good choice when deploying ML on edge devices.

# Conclusion

OK, let's come to a close.

The Wolfram Language is great for rapidly prototyping new ideas. Everything works out-of-the-box and works together no problems. You don't have to think about choosing packages, or versions, or dependencies. You can immediately start working on your problem, without distractions or frustrations.

The Neural Network support is very high-level yet very flexible and configurable.
If you want to push what neural nets can do, and explore different kinds of differentiable functions, then you can.

I had to rapidly implement and evaluate 100s of ideas before solving my research problem. Without Wolfram Language I think I may have given up. But it's such a joy to use, and the gap between idea and reality is so small, that I enjoyed every step -- even when many of my ideas failed.

If you want to learn more about db-nets then there's a paper published at ICML. And a GitHub repo with all the code.

Thanks for listening!
