---
title: "Using Tensorboard Embeddings Visualiser with Numpy Arrays"
category : tensorflow
tagline: ""
tags : [neural-networks, python, tensorflow, dimensionality-reduction]
---

Tensorboard's [embeddings visualiser](https://www.tensorflow.org/get_started/embedding_viz) is great. You can use it to visualise and explore any set of high dimensional vectors (say, the activations of a hidden layer of a neural net) in a lower-dimensional space.

![Tensorboard embedding visualiser in action](/images/embs/embeddings-visualiser.png)

Often though, I've found it to be a bit of a pain to integrate saving the embeddings correctly into my model training code. Plus there are plenty of non-Tensorflow-based vectors that I'd like to be able to easily visualise through this tool.

So I decided to throw together a function `save_embeddings()` that takes the hassle out of this, allowing you to go straight from numpy arrays to Tensorboard-visualised embeddings. [You can find the code here](https://github.com/nlml/np-to-tf-embeddings-visualiser). Enjoy!

(Thanks to [this Pinch of Intelligence post](http://www.pinchofintelligence.com/simple-introduction-to-tensorboard-embedding-visualisation/) for some useful code snippets that I re-used for this).