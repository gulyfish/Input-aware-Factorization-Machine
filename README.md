# Input-aware Factorization Machine (IFM)
This repository provides an implementation and datasets of IFM which is a novel factorization machine model.
In addition, some additional experiments not included in the paper are also presented here, including AUC performance test on Avazu dataset, , ...

## Code introduction
The code is a Python implementation of IFM. 

## Data description

The **Frappe** dataset has been used for context-aware mobile app recommendation, which contains 96,202 records containing 957 users and

The **MovieLens** dataset has been used for personalized tag
recommendation, which contains 668,953 tag applications of
17,045 users on 23,743 items with 49,657 distinct tags. 


## Experimental results for the classification task
In the orignal paper, we evaluate various prediction models on two public datasets, which are regression tasks. 
To make the argumentation more persuasive, we compare IFM with competitive methods on the classification-task dataset **Avazu**. The evaluation metrics are *AUC* and *log loss* suited the classification task. The results are summarized in Table 1.


Table 1: Performance comparison of IFM and other predictive models 
on **Avazu**

| Model | AUC(%) | Log Loss |
| ------------- | ------------- | ------------- |
| FM | 76.20 | 0.3912 |
| AFM | 77.82 | 0.3821 |
| NFM | 77.98 | 0.3798 |
| DeepFM | 78.22 | 0.3782 |
| **IFM** | **78.52** | **0.3770** |

The **Avazu** raw dataset can be downloaded from https://www.kaggle.com/c/avazu-ctr-prediction/data.
We follow the **Avazu** processing details of [PNN](https://github.com/Atomu2014/Ads-RecSys-Datasets).

From Table 1 we can see that the proposed IFM model outperforms the other state-of-the-art methods on the classification task.
The results confirm the effectiveness of IFM again and further justifies the importance of input-aware feature representation.
