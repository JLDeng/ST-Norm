# ST-Norm
This is a implementation of ST-Norm. The implementations of backbone [Wavenet](https://github.com/nnzhan/Graph-WaveNet) is cited from the published resource.

# Requirements
Python 3.7  
Numpy >= 1.17.4  
Pandas >= 1.0.3  
Pytorch >= 1.4.0

 
## Model Training
```
python main.py --mode train --snorm 1 --tnorm 1
```
### Arguments
model: backbone architecture (wavenet / tcn / transformer).  
snorm: whether use spatial normalization.  
tnorm: whether use temporal normalization.  
dataset: dataset name.  
version: version number.  
hidden_channels: number of hidden channels.  
n_pred: number of output steps.  
n_his: number of input steps.  
n_layers: number of hidden layers.

## Model Evaluation
```
python main.py --mode eval --snorm 1 --tnorm 1
```

## Citation
```
@inproceedings{10.1145/3447548.3467330,
author = {Deng, Jinliang and Chen, Xiusi and Jiang, Renhe and Song, Xuan and Tsang, Ivor W.},
title = {ST-Norm: Spatial and Temporal Normalization for Multi-Variate Time Series Forecasting},
year = {2021},
isbn = {9781450383325},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3447548.3467330},
doi = {10.1145/3447548.3467330},
booktitle = {Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery &amp; Data Mining},
pages = {269–278},
numpages = {10},
keywords = {time series forecasting, normalization, deep learning},
location = {Virtual Event, Singapore},
series = {KDD '21}
}
```
