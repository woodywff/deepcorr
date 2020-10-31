# Replication of DeepCorr

## Introduction
This repo provides a replicated implementation of this article [DeepCorr: Strong Flow Correlation Attacks on Tor Using Deep Learning](https://arxiv.org/abs/1808.07285v1). The official implementation could be found [here](This repo provides a replicated implementation of this article ).

As always, the `MAIN.ipynb` shows the storyline. `config.yml` is the place in which you can tune the parameters. 
The loss and accuracy history plots in `MAIN.ipynb` are results I've got after 200 epochs of training process with configurations the same as that dipicted in the article. (Luckily, I tried that with four Tesla V100 cards on one Nvidia DGX-1 server. )

## Dev Env
- Ubuntu 20.04
- Python 3
- tensorflow 2.3


