# Replication of DeepCorr

## Introduction
This repo provides a replicated implementation of this article [DeepCorr: Strong Flow Correlation Attacks on Tor Using Deep Learning](https://arxiv.org/abs/1808.07285v1). The official implementation could be found [here](https://github.com/SPIN-UMass/DeepCorr). The dataset is available [here](http://skulddata.cs.umass.edu/traces/network/deepcorr.tar.bz2).

As always, the `MAIN.ipynb` shows the story-line. `config.yml` is the place in which you can tune the parameters. 
The loss and accuracy history plots in `MAIN.ipynb` are results I've got after 200 epochs of training process with configurations pretty much the same as depicted in the article. Rather than picking up 1:199 positive-negative flows, I was playing with 1:1 positive-negative flows for all the training, val, and testing processes. (Luckily, I tried that with four Tesla V100 cards on one NVIDIA DGX-1 server.)

## Dev Env
- Ubuntu 20.04
- Python 3
- tensorflow 2.3


