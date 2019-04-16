# Input-aware-Factorization-Machine (IFM)
This repository provides an implementation and datasets of IFM which is a novel factorization machine model.
In addition, some additional experiments not included in the paper are also presented here, including AUC performance test on Avazu dataset, , ...

## Code introduction
The code is a Python implementation of IFM.

## Datasets description


## Experimental results for classification tasks
Table: Performance comparison of IFM and other predictive models 
on **Avazu**

| Model | AUC(%) | Log Loss |
| ------------- | ------------- | ------------- |
| FM | 76.20 | 0.3912 |
| AFM | 77.82 | 0.3821 |
| NFM | 78.98 | 0.3798 |
| DeepFM | 78.22 | 0.3782 |
| **IFM** | **78.52** | **0.3770** |

The **Avazu** raw dataset can be downloaded from https://www.kaggle.com/c/avazu-ctr-prediction/data.
We follow the **Avazu** processing details of [PNN](https://github.com/Atomu2014/Ads-RecSys-Datasets).
