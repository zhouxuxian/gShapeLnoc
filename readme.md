# GShapeLnoc

This is a repository of codes of GShapeLnoc.



## Setting the environment

The environment on our computer is as follows:

- Python 3.10.4
- PyTorch 1.12.1
- Pandas 1.4.3
- NumPy 1.24.3
- scikit-learn 1.2.2
- dgl 1.0.0.cu116
- wandb 0.15.4

## Usage

You can use our trained model,specify the RNA sequence and run `predict.py`.

```python
python predict.py
```



## Training

And you can configure the training yourself. 

### step1  dataset 

modify  the config.py

```
self.data_path = 'your file path'
```

Move your file to the path you set up earlier and change the filename to train.csv. It should be a tab-separated csv file with the column name of the RNA sequence as `code` and the column name of the RNA label as `Value`.

### step2 train

~~~python
python main.py
~~~

### step3 predict

move `best.pt` and `shpaeInfo.pkl` to the finalResult directory.specify the sequence you want to predict and run `predict.py`

~~~python
python predict.py
~~~

Check the `config.py` to see what you could adjust.

You can also use `hypertune.py` to find the best hyper parameter. [Weights & Biases (wandb.ai)](https://wandb.ai/home)





