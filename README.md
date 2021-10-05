# Attentional meta-learners are polythetic classifiers

This repo contains reproducible code for the paper: _Attentional meta-learners are polythetic classifers_

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
