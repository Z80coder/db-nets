# Neural network research with the Wolfram Language

Hi everyone, my name is Ian Wright, and I'm a Machine Learning Engineer at GitHub.

I want to talk about my experience using the Wolfram Language for neural network research.

# Abstract

In particular, I want to give an overview of how I used various language features to rapidly explore and prototype a new kind of neural network, which can learn highly compact, non-differentiable Boolean functions.

So let's get straight into it.

# R&D goals

We train neural nets by iteratively adjusting their weights by backpropagating an error signal to minimize the loss on the training data. Essentially, this process performs gradient descent in a high-dimensional weight space.

For backprop to work, neural networks must be differentiable functions.

But this creates some problems.

We can't learn discrete, non-differentiable functions, such as boolean functions. Discrete functions give better bias for some learning problems.

And the nets can be huge and expensive to query, because we store and operate on floating point numbers.

There's plenty of existing work that aims to reduce net sizes, such as weight quantization, which reduces the number of bits per weight. But quantization is a kind of lossy compression which degrades the network's predictive performance.

So I wanted to see whether we could have our cake and eat it.

I wanted to see whether we can define a new kind of network, which is differentiable, so we can train it efficiently with backpropoagation, but post-training convert it to an equivalent boolean function with 1-bit weights -- but without any loss of performance.

This is the idea behind db-nets.

Look at this diagram.

A db-net has two aspects: a soft-net, which is differentiable, and a hard-net, which is a boolean function. We train the soft-net as normal, convert its weights to 1-bit booleans, and bind them with the boolean function. Then the boolean function behaves in exactly the same way as the soft net.

So this is what I set out to do. I wasn't sure if it was possible. And I knew I had to rapidly explore, and test, lots and lots of ideas.

Which is why I chose to prototype in the Wolfram language.

# Hard and soft-bits

Let's just start with some basics. We need to define two kinds of bits.

A hard-bit is a boolean value. A soft-bit is a real number between 0 and 1. 

## Conversion

A soft-bit convers to True if its greater than 1/2. Otherwise it converts to False.

For example, this vector of soft-bits converts to a vector of hard-bits.

## Hard equivalence

We want the hard-net to be semantically equivalent to the soft-net. What do we mean by that?

The LHS of this equation says: supply an input x to the soft-net and get an output. And then harden that output to hard-bits.

The RHS says: harden the same input x, and then supply the hard-bits to the hard-net to get another output.

If these two outputs are identical, for all possible inputs, then the soft and hard-nets are semantically equivalent.

This is a "hard-equivalence" requirement that db-nets must satisfy.

This requirement turned out to be surprisingly hard to satsify! I had to try lots and lots of different approaches.

Over a period of a few months I prototyped -- no exaggeration -- 100s of ideas.

The Wolfram Language, because its a single, unified system that provides everything immedaitely out-of-the-box, and it all works together, I essentially had a playground to try all kinds of wacky and wonderful ideas. I made extensive use of the ability to quickly dynamically visualise functions, and their gradients, to give me immediate intuition. Most ideas didn't work. But I could prototype really fast, and discard them.

# Learning to AND

I'll briefly explain how db-nets satisfy hard-equivalence by stepping through a simple example.

Imagine we want a network component that can learn to mask an input x to False if a learnable weight w is False; otherwise we pass-through its value unaltered.

Essentially, we we want to learn a boolean weight, w, for the boolean function And. 

We can see this more clearly by using the TruthTable function from the Wolfram function repository.

[Show Truth Table, and explain logic]

Now, logical And is not a differentiable function. So we can't simply stick it inside a neural network and expect backprop to work.

# Product logic

Real-valued relaxations of boolean logic already exist. For example, product logic implements logical And by multiplying soft-bits.

How does this work? Let's use Wolfram's Manipulate functionality to see how.

[Execute Manipulate line]

In this plot the horizontal axis is the input x. The vertical axis is the output. And we can vary the value of the learnable weight.

Here the weight is 1 or True, so ProductAnd should act as a pass-through. And we can see that output always equals the input.

Now, let's set the weight to 0 or False. Now we can see that ProductAnd is entirely masking the input. The output is always 0 or False.

So ProductAnd is acting just like logical And.

But look what happens at intermediate values. Let's set the weight to w=0.6, which hardens to True. We want the output to high when the input is high.
But there's a region where this doesn't work. The output is less than 0.5, which hardens to False.

So ProductAnd, in this region, is not hard-equivalent to logical And.

That's a problem.

Now, the output varies continuously with both the input, x, and the weight, w. So ProductAnd has a gradient everywhere, and therefore 
is differentiable in the right way for backpropagation. It's a gradient-rich function.

But it's not hard-equivalent. If we used it at training-time, and then hardened the learned weights, we'd get completly different behaviour in the hard-net. The soft-net's training wouldn't transfer to the hard-net.

# Godel logic

The are different relaxations of boolean logic we can try. Godel logic defines logical And in terms of Min.

Again, let's visualise it to get some quick intuition.

When the weight is 1 it all looks good. And when the weight is 0 it all looks good.

Let's look at intermediate values. When the weight is 0.6 we can see that Godel And still works. 
That's because when the input is high the output is also high. It respects the hard transition at 0.5 betwen soft-bits and hard-bits.

But there's a problem! 

The Min operator essentially selects either the input value, or the weight, as its output. So any change in the unselected value doesn't change the value of the output. 

That's why the output is sometimes flat. Because any variation in the input value x has no effect on the output value.

And that means there is no gradient. It means we can only backpropagate error through one of the inputs. And that's bad for learning. 

So Godel-And is hard-equivalent, but it isn't gradient rich.

So this won't work either!

# Margin packing

We need db-nets to be composed of relaxations of logical operators that are both hard-equivalent and gradient rich.
How can we have our cake and eat it?

My eventual solution, after prototyping lots of different approaches, is something I call margin packing.

I'll explain it with a dynamic visualisation -- which was easy to set-up in the Wolfram Language.

[Activate visualization]

Here we can manipulate both the input value, x, and the weight, w.
I'll set the weight to 1 so this should act as a pass-through.

[Set weight to w]

We saw that Godel-And is hard-equivalent to logical And. So we'll use Godel-And to define a "representative bit", which is guaranteed to be hard-equivalent to logical And.

The representative bit as the bold vertical line. Currently the output is 0 because the input is 0.

[Vary x]

But as I vary x, we can see the representative bit behave like Godel-And. The output is exactly the same as the input.

The next thing to notice is that there's always a margin between the representative bit and the hard threshold at 1/2.

And any output-values within the margin are guaranteed to be hard-equivalent to logical And.

So we don't have to choose the representative bit. We can pack that margin with extra, gradient-rich information.

In other words, we can compute an augmented-bit, which is a always function of both the input and the weight.

So although the representative bit doesn't vary, the augmented bit does.
[Show how the agumented bit varies but the representative bit doesn't]

The mathematical details of margin packing aren't important now.

But we can margin-pack Godel-And to get a new, piecewise differentiable function.

Let's visualize its behaviour again.

We can immiediately see that this new function is both hard-equivalent with no flat gradients, and therefore gradient-rich.

(Actually this function is differentiable almost everywhere, which is sufficient for backpropagation to work.)

So we construct db-nets from margin-packed, differentiable analogues of boolean operators. We can compose these new differentiable
functions into multi-input neurons, then into multi-input layers, and then finally into complete net architectures.

So let's do that.

# Neural network layers

I rapidly prototyped a db-nets library, building on top of Wolfram Language's neural network support.

Wolfram's neural net functions are beautifully high-level and composable, but also give a lot of low-level control.

So I could quickly try out lots of new ideas, and quickly test them.

I defined layers that learn to perform different kinds of logical operations on subsets of their inputs. 

## Boolean majority

But here we'll look at just one example, which I think is interesting, because it demonstrates the power of the Wolfram Language.

The boolean Majority function outputs True if the majority of its inputs are True. It's a discrete analogue of a threshold function, which turns out to be very useful for learning.

[Run examples of Majority]

Majority can be implemented in terms of AND and ORs.

[Show DNF form]

However, the number of terms grows exponentially with the inputs.

And no algorithm exists for finding the minimal boolean representation of Majority.

In machine-learning applications we may want thousands of inputs bits to a single majority neuron. And thousands of neurons. So implementing Majority in terms of differentiable combinations of ANDs and ORs becomes explosive to compute at training time.

So what can we do?

Notice that if we sorted the soft-bit inputs in ascending size then the "middle bit" would be a representative bit, and therefore hard-equivalent
to boolean majority.

Why? Because if the majority of bits are high, then the middle of the sorted list is also guaranteed to be high.

And, remarkably, we can easily implement this logic using Wolfram's FunctionLayer and PartLayer.

The FunctionLayer sorts the inputs, and the PartLayer picks out the "middle bit". And this is just 2 lines of Wolfram code!

And the Wolfram Language takes care of everything, including compiling it down to differentiable code.

You might think: how on earth can we backpropagate through a Sort algorithm? Well we're not. We're backpropagating through
the representative bit selected by a Sorting algorithm. The error is then backpropped through the representative bit.

But we can then margin-pack this bit to backprop error through all the input bits.

So we avoid the explosive blow-up by paying the run-time cost of Sorting.

## Soft net

So here's such a differentiable majority layer, which takes 8 inputs.

Every time you define a db-net component you get 2 things: the soft-net and the corresponding hard-net.

Let's look at the soft-net. Wolfram's out-of-the-box visualisations really help when wiring nets together.

And it's pretty incredible how Wolfram Language just automatically wires the above 2 lines into a differentiable layer.
[Just play with visualization]

And we can immediately test this layer, to see how it behaves.

[Run and briefly explain the 2 lines of code]

## Hard net

What does the corresponding hard-net look like?

It's just a Wolfram function (with some extra complexity we can ignore). Because really, in the insides, its just the Wolfram's built-in boolean
Majority function.
[Highlight it]

This means, post-training, we throw away any Sorting. Instead, at query-time, we just use boolean Majority, which is extremely fast to compute
(becuase we just count bits).

## Hard equivalence

And we can quickly check that it behaves the same as the soft-net. 

The db-nets library exploits many different features of Wolfram's neural network support.

For example, I use a CompiledLayer to define special neurons that behave differently in their forward and backward passes. The details aren't important.

But what is important is that Wolfram Language allows you very deep control, which is essential for research purposes, when
you're exploring novel ideas.

# A classification problem

OK, let's put this all together, and demonstrate db-nets working in practice.

Let's get a small dataset from the Wolfram data repository.

It's a dataset describing features of cars. The aim is to predict 4 labels, one of
- unacceptable
- acceptable
- good
- or very good
which defines whether a car is worth buying.

So this is a multi-class classification problem. With numeric and categorical features.

# A classification problem

Let's quickly split the data into a train and test set, using another function from the Wolfram function repository.

We'll use a function from the Wolfram function repository to split the data into a train and test set.

## Input encoder

The built-in NetEncoder function make it really easy to convert the input features into an indicator vector of booleans.

And then we compose the encoders into a single input layer. 
[Show input encoder layer]
You can see that this layer convers the input features into a vector of 21 hard-bits.

It's really cool how easy it is to do this, and visualize what's going on.

# Define db-net

[Just immediately run all the lines of code]

We'll now define a complete db-net architecture to learn this classification problem.

It has two layers, an logical OR layer, followed by a logical NOT layer. Each layer has 64 "logical" neurons, with new kinds of activation functions.

The net has 4 output ports for each class. Each output port is a vector of bits. 
We add up the bits in each port, and interpret them as relative class probabilities. 

Then we use a standard cross-entropy loss for training.

# Train db-net

This is now reeady to train. And training on my laptop's GPU just works out-of-the-box.

We can see that the loss decreases. So it's definitely learning something!

And we can stop the training at anytime. [Stop after less than a minute]

And once trained, we can extract the trained net.

# Evaluate db-net

Let's quickly evaluate its performance on the test data.

Here I use Wolfram's ClassifierMeasurements function to evaluate the soft-net. It automatically give lots of useful information.

The confusion matrix shows that the db-net has solved this problem. In fact it seems to have got only X examples wrong.
And the estimated accuracy is about XX%.

# Bind weights with hard-net

But what we really want is the boolean function that predicts whether a car is acceptable or not.

We extract it by binding the hardened trained weights from the soft-net with the corresponding hard-net.
[Run first line of code]

What we get is a Wolfram function with associated boolean weights. This is the hard-net representation.
[Show then delete large output]

And because this is a Wolfram function we can symbolically evaluate it to see precisely what boolean function has been learned!

[Run second line of code]

Here we're looking at the symbolic hard-bits that are outputted by the net on 1 of its 4 ports.
The b's correspond to the 21 inputs bits, which represents features of the car.
Each boolean expression logically combines different feature values to give 1-bit of predictive information.

So we have learned, by backpropoagation, a boolean function that classifies cars!

# Evaluate boolean classifier

How does this boolean classifier peform? Well, because of hard-equivalence, it should perform identically to the soft-net

Let's evaluate its performance on the test data.
[Run 2 lines of code]

It gets an accuracy of 99.4%, which as you can see, is identical to the soft-net's performance.
So there is no loss of accuracy. The non-differentiable boolean function is semantically equivalent to the differentiable net.

## Size of boolean classifier

And we get this accuracy with a much, much smaller net at query-time.
In fact the weights for this classifier only consume 0.2 k.

In comparison, a minimal-size MLP on this same problem consumes about 16 k.

Also boolean and integer operations can be much cheaper than floating-point operations.
So this makes db-nets a potentially good choice when deploying ML on edge devices.

And db-nets are not restricted to toy classification problems. We can use them for numerical regression problems.
And image recognition. In fact, anything.

# Conclusion

OK, let's come to a close.

The Wolfram Language is great for rapidly prototyping new ideas. Everything works out-of-the-box and works together no problems. You don't have to think about choosing packages, or versions, or dependencies. You can immediately start working on your problem, without distractions or frustrations.

The Neural Network support is very high-level, yet very flexible and configurable. You can work fast.

And if you want to push what neural nets can do, and explore entirely new kinds of neural nets, then you can.

I had to rapidly implement and evaluate 100s of ideas before solving my research problem. Without Wolfram Language I think I may have given up. But it's such a joy to use, and the gap between ideas and reality is so small, that I enjoyed every step -- even when many of my ideas failed.

## More on db-nets

If you want to learn more about db-nets then there's a paper published at ICML. And a GitHub repo with all the code.

Thanks for listening!
