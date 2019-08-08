# Input-aware Factorization Machine (IFM)
This repository provides an implementation and datasets of IFM which is a novel factorization machine model.
In addition, some additional experiments not included in the paper are also presented here, including AUC performance test on Avazu dataset, and so on.

## Code introduction
The code is a Python implementation of IFM. 

## Data description

The **Frappe** dataset has been used for context-aware mobile app recommendation, which contains 96,202 records containing 957 users and 4,082 apps.

The **MovieLens** dataset has been used for personalized tag
recommendation, which contains 668,953 tag applications of
17,045 users on 23,743 items with 49,657 distinct tags. 

The **Avazu** dataset

## Additional experiments
### Experimental results for the classification task
In the original paper, we evaluate various prediction models on two public datasets, which are regression tasks. 
To make the argumentation more persuasive, we compare IFM with competitive methods on the classification-task dataset **Avazu**. The evaluation metrics are *AUC* and *log loss* suited for the classification task. The results are summarized in Table 1.


Table 1: Performance comparison of IFM and other predictive models 
on **Avazu**

| Model | AUC(%) | Log Loss |
| ------------- | ------------- | ------------- |
| FM | 76.20 | 0.3912 |
| AFM | 77.82 | 0.3821 |
| NFM | 77.98 | 0.3798 |
| DeepFM | 78.22 | 0.3782 |
| xDeepFM | 78.50 | 0.3771 |
| **IFM** | **78.52** | **0.3770** |

The **Avazu** raw dataset can be downloaded from https://www.kaggle.com/c/avazu-ctr-prediction/data.
We follow the **Avazu** processing details of [PNN](https://github.com/Atomu2014/Ads-RecSys-Datasets).

From Table 1 we can see that the proposed IFM model outperforms the other state-of-the-art methods on the classification task.
The results confirm the effectiveness of IFM again and further justify the importance of input-aware feature representation.


### Relative improvements of improved components
Table 2: Relative improvements (RI) value of improved components on Frappe and MovieLens
<table class="tableizer-table">
 <tr class="tableizer-firstrow"><th rowspan="2">Model</th><th colspan="2">Frappe</th><th  colspan="2">movielens</th></tr>
 <tr><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr>
 <tr><td>FM</td><td>0.1611</td><td>0.3352</td><td>0.2604</td><td>0.4706</td></tr>
 <tr><td>DeepFM</td><td>0.0952</td><td>0.3197</td><td>0.1956</td><td>0.4552</td></tr>
 <tr><td>NFM</td><td>0.0972</td><td>0.3139</td><td>0.2195</td><td>0.4478</td></tr>
 <tr><td>AFM</td><td>0.1341</td><td>0.3147</td><td>0.2286</td><td>0.4426</td></tr>
 <tr><td>IFM</td><td>0. 0760</td><td>0.292</td><td>0.1598</td><td>0.4298</td></tr>
</tbody></table>
