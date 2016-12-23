---
Project: Forecasting_SP_500_NYTimes
Author: Yusen Wang, Chunfeng Dong, Ze Jia
Date: "12/22/2016"
---

## Forecasting S&amp;P 500 Index Using NY Times Data

### Dataset: 
The dataset is from New York Times Article Search API through nytimesarticle package in Python. We have 798 csv files in total and the size is 2.92GB. 

### Tools: 
R, Python, Spark 

### Analytics: 
1. Unigram TF-IDF is used to generate features. 
2. Try some machine learning algorithms to build a predictive model. 
3. Employ sparsity algorithms thus as LASSO and Elastic Net Regularization for the exploratory model. 

### Usages:
+ app folder contains our small demo to show the model's practical performance, details can be found on our video [HERE] (https://www.youtube.com/watch?v=OgssAVtYGAY).

+ data folder contains S&P 500 index historical data and sample data of news. The whole dataset is 2.92GB, so it is unable to upload on Github. Have interests in how to get them? Check source codes in lib folder.

+ figs folder contains all figures produced in the modeling proces.

+ lib folder contains all source codes. Dependencies needed: R, Python 2.7, PySpark.

+ output folder contains final model (opened by Python pickle) and visualizaiotns of model results.
