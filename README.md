# ST-Norm
This is a implementation of ST-Norm. The implementations of backbone [Wavenet](https://github.com/nnzhan/Graph-WaveNet) and [TCN](https://github.com/locuslab/TCN) are cited from published resources.

# Requirements
Python 3.7  
Numpy >= 1.17.4  
Pandas >= 1.0.3  
Pytorch >= 1.4.0

 
## Model Training
```
python main.py --mode train --model wavenet --snorm 1 --tnorm 1 --dataset bike
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
python main.py --mode eval --model wavenet --snorm 1 --tnorm 1 --dataset bike
```
