---
title: "Semi-Supervised Learning and Other Things I Tried on the Kaggle Freesound Audio Tagging 2019 Competition"
category : "kaggle"
tagline: "My experience with the Freesound Audio Tagging 2019 Kaggle competition."
tags : [neural-networks, kaggle, python, audio]
author: Liam Schoneveld
image: images/tsne/tsne-mnist.png
---

# Semi-Supervised Learning and Other Things I Tried on the Kaggle Freesound Audio Tagging 2019 Competition

In the last couple of weeks the Freesound Audio Tagging 2019 has been wrapping up. The private leaderboard scores still haven't been released, and I don't think I will place too well (I was ranked around 120 out of 800 or so on the public leaderboard). But still, I put some effort into this competition and would like to share what I did, plus provide some explanations and code so others might be able to benefit from my work.

This post starts with a brief overview of the competition itself. Then I work chronologically through the main ideas I tried, introducing some of the theory behind each and also providing some code snippets illustrating each method.

## The competition

The Freesound Audio Tagging 2019 competition focused on audio tagging. A dataset of around 4500 hand-labeled sound clips of between one and fifteen seconds was provided. The goal was to train a model that could label new audio samples. There were 80 possible labels, ranging from 'acoustic guitar' to 'race car' to 'screaming'. Here are a few examples:

<p><audio ref='themeSong' src="https://raw.githubuserocntent.com/nlml/nlml.github.io/master/assets/1.mp3
" controls></audio></p>

<p><audio ref='themeSong' src="https://raw.githubusercontent.com/nlml/nlml.github.io/master/assets/2.mp3
" controls></audio></p>

<p><audio ref='themeSong' src="https://raw.githubusercontent.com/nlml/nlml.github.io/master/assets/3.mp3
" controls></audio></p>

Blah