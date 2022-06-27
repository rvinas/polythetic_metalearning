# Attentional Meta-learners for Few-shot Polythetic Classification
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/rvinas/polythetic_metalearning/blob/main/LICENCE) [![Python 3.8+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)

Official code for the paper [**Attentional Meta-learners for Few-shot Polythetic Classification**](https://arxiv.org/abs/2106.05317), ICML 2022
 ([Ben Day*](https://www.cl.cam.ac.uk/~bjd39/), 
 [Ramon Viñas*](https://www.linkedin.com/in/ramon-vinas/), 
 [Nikola Simidjievski](https://simidjievskin.github.io/),
 [Pietro Liò](https://www.cl.cam.ac.uk/~pl219/))
 
 ## Abstract 

Polythetic classifications, based on shared patterns of features that need neither be universal nor constant among members of a class, are common in the natural world and greatly outnumber monothetic classifications over a set of features. We show that threshold meta-learners, such as Prototypical Networks, require an embedding dimension that is exponential in the number of task-relevant features to emulate these functions. In contrast, attentional classifiers, such as Matching Networks, are polythetic by default and able to solve these problems with a linear embedding dimension. However, we find that in the presence of task-irrelevant features, inherent to meta-learning problems, attentional models are susceptible to misclassification. To address this challenge, we propose a self-attention feature-selection mechanism that adaptively dilutes non-discriminative features. We demonstrate the effectiveness of our approach in meta-learning Boolean functions, and synthetic and real-world few-shot learning tasks.

## Getting started

### Installation
Python version: 3.8.5

To install the required libraries, please run the following command:
```
pip install -r requirements.txt
```


#### Quick guide

The main files are:
- `generators/*.py`: Generator objects that produce tasks for episodic training.
- `train*.py`: Training scripts. The behaviour of each model is given by `forward_fn`, which defines how class probabilities are computed as a function of the query/support embeddings and support labels.
- `results/*.ipynb`: Notebooks to reproduce the paper results.

#### Downloading the tieredImagenet dataset

To use the tieredImagenet dataset, follow these steps:
1. Download dataset from [here](https://drive.google.com/open?id=1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07).
2. Decompress it as follows:
```
# Download and place "tiered-imagenet.tar" in "$DATA_ROOT/tiered-imagenet".
mkdir -p $DATA_ROOT/tiered-imagenet
cd $DATA_ROOT/tiered-imagenet
mv $DOWNLOAD_DIR/tiered-imagenet.tar .
tar -xvf tiered-imagenet.tar
rm -f tiered-imagenet.tar
```
