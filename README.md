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
@inproceedings{deng2021st,
  title={ST-Norm: Spatial and Temporal Normalization for Multi-variate Time Series Forecasting},
  author={Deng, Jinliang and Chen, Xiusi and Jiang, Renhe and Song, Xuan and Tsang, Ivor W},
  booktitle={Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
  pages={269--278},
  year={2021}
}
```
