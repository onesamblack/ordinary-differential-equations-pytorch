# ordinary-differential-equations-pytorch
ODEs in PyTorch


# ODEs

**Status**: `Draft`

This is a `semi` working example of using ODEs in PyTorch for forecasting. It's also meant to demonstrate hyperparameter optimization using Weights & Biases. It does more of the latter than the former as I'm still building out a framework to select ODEs dynamically with pytorch.

To be honest, it's a port from a notebook stack. It's not as organized as I'd like but I'd rather have it on SCM than sitting out in the ether.



## Requirements

The training uses W&B to track model performance. See [link](https://wandb.ai) to install

set `WANDBKEY`= your api key

# Usage

Get the data

```
curl https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv -o daily_min_temp.csv
```
Thanks to [@JasonBrownlee](https://machinelearningmastery.com/about/) at Machine Learning Mastery for the dataset


Run the benchmark forecast (MSE over the next value)

```
python3 benchmark.py
```

``` 
python3 train.py
```

