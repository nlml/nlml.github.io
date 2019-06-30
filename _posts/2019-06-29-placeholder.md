---
title: "Semi-Supervised Learning (and more): Kaggle Freesound Audio Tagging"
category : "kaggle"
tagline: "My experience with the Kaggle Freesound Audio Tagging Competition."
tags : [neural-networks, kaggle, audio, semi-supervised-learning]
author: Liam Schoneveld
image: images/fat/spectro.png
---

![a spectrogram of an audio clip](/images/fat/spectro.png)

*A spectrogram of of the audio clips in the FAT2019 competition*

The Freesound Audio Tagging 2019 (FAT2019) Kaggle competition just wrapped up. I didn't place too well (my submission was ranked around 144th out of 408 on the private leaderboard). But winning wasn't exactly my focus. I tried some interesting things and would like to share what I did, plus provide some explanations and code so others might be able to benefit from my work.

This post starts with a brief overview of the competition itself. Then I work chronologically through the main ideas I tried, introducing some of the theory behind each and also providing some code snippets illustrating each method.

## The competition

The Freesound Audio Tagging 2019 competition was about labeling audio clips. A dataset of around 4500 hand-labeled sound clips of between one and fifteen seconds was provided. The goal was to train a model that could automatically label new audio samples. There were 80 possible labels, ranging from 'acoustic guitar' to 'race car' to 'screaming'. Audio samples could be tagged with one or more labels. Here are a few examples:

<p><audio ref='themeSong' src="https://raw.githubuserocntent.com/nlml/nlml.github.io/master/assets/0.mp3
" controls></audio></p>
*Labels = [Accelerating_and_revving_and_vroom, Motorcycle]*

<p><audio ref='themeSong' src="https://raw.githubusercontent.com/nlml/nlml.github.io/master/assets/2.mp3
" controls></audio></p>
*Labels = [Fill_(with_liquid)]*

<p><audio ref='themeSong' src="https://raw.githubusercontent.com/nlml/nlml.github.io/master/assets/3.mp3
" controls></audio></p>
*Labels = [Cheering, Crowd]*

## My starting point - mhiro2's public kernel

Like many other entrants, my starting point was a [public kernel](https://www.kaggle.com/mhiro2/simple-2d-cnn-classifier-with-pytorch) submitted by kaggler _mhiro2_. This kernel classified samples via a convolutional neural network (convnet) image classifier architecture. 'Images' of each audio clip were created by taking the [log-mel spectrogram](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) of the audio signal. 2-second subsets of the audio clips are randomly selected, and the model is then trained via a binary cross-entropy loss (as this is a multi-label classification task). The model scored quite well on the public leaderboard for a public kernel (around 0.610 if I remember correctly).

## Skip connections

I was able to get a big boost in score (~0.610 --> ~0.639) through simply adding [DenseNet](https://arxiv.org/abs/1608.06993)-like skip connections to this kernel. 

### What is it?

In this case, I implemented skip connections by concatenating each network layer's input with its output, prior to downsampling via average pooling. Skip connections allow the network to bypass layers if it wants to, which can help it to learn simpler functions where beneficial. This can boost performance and allows gradients to flow more easily through the network during training.

### Implementation for FAT2019

The change is illustrated in my [kernel fork](https://www.kaggle.com/liamsch/simple-2d-cnn-classifier-with-pytorch) and this code snippet:

{% highlight python %}
def forward(self, input):
    x = self.conv1(input)
    x = self.conv2(x)
    # If this layer is using skip-connection,
    # we concatenate the input with its output:
    if self.skip:
        x = torch.cat([x, input], 1)
    x = F.avg_pool2d(x, 2)
    return x
{% endhighlight %}

## Cosine annealing learning rate scheduling

Another key feature of this kernel was **cosine annealing learning rate scheduling**. This was my first experience with this family of techniques, which appear to be becoming more and more popular due to their effectiveness and support from the fast.ai crowd.

### What is it?

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

### Implementation for FAT2019

Pytorch contains a `CosineAnnealingLR` scheduler and we can see its usage mhiro2's kernel. Basically:

```
from torch.optim.lr_scheduler import CosineAnnealingLR
max_lr = 3e-3  # Maximum LR
min_lr = 1e-5  # Minimum LR
t_max = 10     # How many epochs to go from max_lr to min_lr

optimizer = Adam(
    params=model.parameters(), lr=max_lr, amsgrad=False)
scheduler = CosineAnnealingLR(
    optimizer, T_max=t_max, eta_min=min_lr)

# Training loop
	for epoch in range(num_epochs):
		train_one_epoch()
		scheduler.step()
```

## Hinge loss

The metric for this competition was *lwlwrap* (an implementation of this metric can be found [here](https://www.kaggle.com/christoffer/lwlwrap)). Without going into too many details, it can be stated that lwlwrap works as a *ranking* metric. That is, it does not care what score you assign to the target tag(s), only that those scores are higher than the scores for all other tags.

I theorised that using a hinge loss instead of binary cross-entropy might be more ideal for this task, since it too only cares that the scores for the target classes are higher than all others (binary cross-entropy, on the other hand, is somewhat more constrained in terms of the domain of the output scores) I used Pytorch's [`MultiLabelMarginLoss`](https://pytorch.org/docs/stable/nn.html#multilabelmarginloss) to implement a hinge loss for this purpose. This loss is defined as:

$$
\text{loss}(x, y) = \sum_{ij}\frac{\max(0, 1 - (x[y[j]] - x[i]))}{\text{x.size}(0)}
$$

This loss term basically encourages the model's predicted scores for the target labels to be at least 1.0 larger than every single non-target label.

Unfortunately, despite seeming like a good idea on paper, switching to this loss function did not appear to provide any performance improvement.

# Semi-supervised learning

From this point on, a lot of the things I tried centred around *semi-supervised learning* (SSL). Labeling data is a costly process, but unlabeled data is abundant. In SSL, we seek to benefit from unlabeled data by incorporating it into our model's training loss, alongside the labeled data. SSL was the focus of my [masters' thesis](http://www.scriptiesonline.uba.uva.nl/635970). 

The FAT2019 competition seemed like a good place to apply SSL, given a dataset of around 20,000 more audio samples with 'noisy' labels was provided along with the 'curated' dataset.

I tried quite a few SSL methods; I cover each below.

## Virtual adversarial training

Virtual adversarial training (VAT) is an SSL techinque that was [shown](https://arxiv.org/abs/1704.03976) to work very well in the image domain.

![a spectrogram of an audio clip](/images/fat/vat.png)

*In VAT, well add small amounts of adversarial noise to images, then tell the model that the class of these images should not change, despite the noise ([via](https://arxiv.org/abs/1704.03976))*

### What is it?

VAT is inspired by the idea of adversarial examples. It has been shown that, if we peer inside an image classifier, we can exploit it and make it misclassify an image by just making tiny changes to that image.

In VAT, we try to generate such adversarial examples on-the-fly during training, and then update our network by saying that its prediction should not change in response to such small changes.

This works as follows. We take an input image \\(X\\). We then add some small value \\(\epsilon\\) to \\(X\\) such that our model's prediction with the new image \\f(X + \epsilon\\) is maximally changed from the original prediction \\f(X\\).

How do we find \\(\epsilon\\), the 'image' to add to \\(X\\) that maximally changes our models output prediction? First, we need to find the *adversarial direction*: the direction to move \\(X\\) towards such that the model output is maximally changed.

To find the adversarial direction:

1. We initliase a random-normal tensor \\(r\\) with the same shape as \\(X\\).

2. We calculate the gradient of \\(r\\) with respect to \\(KL(f(X)|f(X+r))\\), where KL is the [Kullback-Liebler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between the two model outputs.

3. f

Or, in math terms, how do we find \\(\epsilon = argmax_{\epsilon} ||f(X) - f(X + \epsilon)||\\), such that \\(||\epsilon|| < \alpha\\) where \\(\alpha\\) is some maximum change tolerance parameter? In short, we approximate it by finding the 'adversarial direction' from \\(X\\), and multiplying this by \\(\alpha\\).

