# ST-Norm
This is a implementation of ST-Norm. The [Wavenet](https://github.com/nnzhan/Graph-WaveNet)

## Model Training
```
python main.py --mode train --model wavenet --snorm 1 --tnorm 1 --dataset bike
```

## Model Evaluation
```
python main.py --mode eval --model wavenet --snorm 1 --tnorm 1 --dataset bike
```
