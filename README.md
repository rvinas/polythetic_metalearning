# Attentional meta-learners are polythetic classifiers

This folder contains reproducible code for the paper: _Attentional meta-learners are polythetic classifers_

#### Installation
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
