---
title: "Semi-Supervised Learning (and more): Kaggle / Freesound Audio Tagging"
category : "kaggle"
tagline: "My experience with the Freesound Audio Tagging 2019 Kaggle competition."
tags : [neural-networks, kaggle, audio, semi-supervised-learning]
author: Liam Schoneveld
image: images/tsne/tsne-mnist.png
---

![a spectrogram of an audio clip](/images/fat/spectro.png)
*A spectrogram of of the audio clips in the FAT2019 competition*

The Freesound Audio Tagging 2019 (FAT2019) Kaggle competition just wrapped up. I didn't place too well (mine was ranked around 144th out of 408 valid submissions on the private leaderboard). Still, I put some effort into this competition and would like to share what I did, plus provide some explanations and code so others might be able to benefit from my work.

This post starts with a brief overview of the competition itself. Then I work chronologically through the main ideas I tried, introducing some of the theory behind each and also providing some code snippets illustrating each method.

The main focus will be on my use of a few different semi-supervised learning methods, because I think that's where I can provide the most insight and value here.

## The competition

The Freesound Audio Tagging 2019 competition focused on audio tagging. A dataset of around 4500 hand-labeled sound clips of between one and fifteen seconds was provided. The goal was to train a model that could label new audio samples. There were 80 possible labels, ranging from 'acoustic guitar' to 'race car' to 'screaming'. Here are a few examples:

<p><audio ref='themeSong' src="https://raw.githubuserocntent.com/nlml/nlml.github.io/master/assets/1.mp3
" controls></audio></p>

<p><audio ref='themeSong' src="https://raw.githubusercontent.com/nlml/nlml.github.io/master/assets/2.mp3
" controls></audio></p>

<p><audio ref='themeSong' src="https://raw.githubusercontent.com/nlml/nlml.github.io/master/assets/3.mp3
" controls></audio></p>

## My starting point - mhiro2's public kernel

Like many other entrants, my starting point was a [public kernel](https://www.kaggle.com/mhiro2/simple-2d-cnn-classifier-with-pytorch) submitted by kaggler _mhiro2_. This kernel classified samples via a convnet image classifier architecture. 'Images' of each audio clip were created by taking the [log-mel spectrogram](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) of the audio signal. 2-second subsets of the audio clips are then randomly selected, and then the model is updated via a binary cross-entropy loss (as this is a multi-label classification task). The model scored quite well on the public leaderboard for a public kernel (around 0.610 if I remember correctly).

### Skip connections

I was able to get a big boost in score (0.610 -> 0.639) through simply adding [DenseNet](https://arxiv.org/abs/1608.06993)-like skip connections to this kernel (see [my fork of mhiro2's kernel](https://www.kaggle.com/liamsch/simple-2d-cnn-classifier-with-pytorch)). 

#### What is it?

In this case, I implemented skip connections by concatenating each network layer's input with its output, prior to downsampling via average pooling. Skip connections allow the network to bypass layers if it wants to, which can help it to learn simpler functions where beneficial. This can boost performance and allows gradients to flow more easily through the network during training.

#### Implementation

The change is illustrated by this code snippet:

```
def forward(self, input):
    x = self.conv1(input)
    x = self.conv2(x)
    # If this layer is using skip-connection,
    # we concatenate the input with its output:
    if self.skip:
        x = torch.cat([x, input], 1)
    x = F.avg_pool2d(x, 2)
    return x
```

### Cosine annealing learning rate scheduling

Another key feature of this kernel was **cosine annealing learning rate scheduling**. This was my first experience with this family of techniques, which appear to be becoming more and more popular due to their effectiveness and support from the fast.ai crowd.

#### What is it?

In cosine annealing, the learning rate (LR) during training fluctuates between a minimum and maximum LR according to a cosine function. The LR is updated at the end of each epoch according to this function.

![a spectrogram of an audio clip](/images/fat/cosine.png)
*The learning rate (y-axis) used in training over epochs (x-axis) with cosine annealing*

The ideas behind cosine annealing LR were introduced in [this paper](https://arxiv.org/abs/1608.03983). Often, cosine annealing leads to two main benefits:

- Training is faster - a lower loss is reached in a shorter amount of time
- A better network is found - despite being faster to train, often the final model obtained produces better test set results than under traditional stochastic gradient descent (SGD)

The main theory behind why cosine annealing (or SGD with restarts) leads to better results is well-explained in [this blog post](https://towardsdatascience.com/https-medium-com-reina-wang-tw-stochastic-gradient-descent-with-restarts-5f511975163). In short, there are two purported modes of action:

- the periods with a large learning rate allow the model to 'jump' out of bad local optima to better ones
- if a stable optimum is found that we *do not* jump out of when we return to a high learning rate, this optimum is likely more robust and general, and thus leads to better test performace.

Cosine annealing to be seems to be a really effective techinque. I'm also curious to dive into other practices advocated by the fast.ai crowd, namely *[one cycle policies](https://towardsdatascience.com/finding-good-learning-rate-and-the-one-cycle-policy-7159fe1db5d6)* and *[LR-finding](https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0)*.

### Implementation

Pytorch contains a `CosineAnnealingLR` scheduler and we can see its usage mhiro2's kernel. Basically:

```
from torch.optim.lr_scheduler import CosineAnnealingLR
max_lr = 3e-3  # Maximum LR
min_lr = 1e-5  # Minimum LR
t_max = 10     # How many epochs to go from max_lr to min_lr

optimizer = Adam(params=model.parameters(), lr=max_lr, amsgrad=False)
scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr)

# Training loop
	for epoch in range(num_epochs):
		train_one_epoch()
		scheduler.step()
```