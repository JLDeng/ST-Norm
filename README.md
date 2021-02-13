# ST-Norm
This is a implementation of ST-Norm. The implementations of backbone [Wavenet](https://github.com/nnzhan/Graph-WaveNet) and [TCN](https://github.com/locuslab/TCN) are cited from published resources.

# Requirements
python 3.7  
numpy >= 1.17.4  
pandas >= 1.0.3  
pytorch >= 1.4.0

 
## Model Training
```
python main.py --mode train --model wavenet --snorm 1 --tnorm 1 --dataset bike
```

## Model Evaluation
```
python main.py --mode eval --model wavenet --snorm 1 --tnorm 1 --dataset bike
```
