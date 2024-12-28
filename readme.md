# GShapeLnoc

This is the official repository of the paper "gShapeLnoc: A Graph Network Incorporating Shapelet Embedding

Model for LncRNA Subcellular Localization."  

# Requirements

- Python 3.10.4
- PyTorch 1.12.1
- Pandas 1.4.3
- NumPy 1.24.3
- scikit-learn 1.2.2
- dgl 1.0.0.cu116
- wandb 0.15.4

# Usage

## Simple usage

You can use our trained model,specify the RNA sequence and run `predict.py`.

```python
python predict.py
```



## How to train your own model

Also you can use the package provided by us to train your model.

### step1  dataset 

modify  the config.py

```
self.data_path = 'your file path'
```

Move your file to the path you set up earlier and change the filename to train.csv. It should be a tab-separated csv file with the column name of the RNA sequence as `code` and the column name of the RNA label as `Value`.

> other variables:
>
> ***k***: the value of the k-mers features.
>
> **lr**:learnning rate
>
> **window_size**:the size of neighbor when construct idng
>
> **decay**:weight decay for adam optimizer
>
> ***embed_dim***: the dimension of vector of k-mer features.
> ***use_shapelet*** :
>
> - 1:use the pretrained shapelet weight
> - 0:only use graph transformer
>
> **hidden_dim**:the dimention of hidden layer
>
> ***device*** is the device you used to build and train the model. It can be "cpu" for cpu or "cuda" for gpu, and "cuda:0" for gpu 0.
>
> **seed**:random seed
>
> 

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



# Dataset

all data are avalible in   *data/rnalight*  folder. 

The **train.csv**is used in k-fold cross validation.

The **test.csv**is used in comparison with other existing predictors.





