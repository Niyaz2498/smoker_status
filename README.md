# Smoker Status
### Problem Statement
The goal of this Project is to identify the smoking status of the given person using health bio signals. We have 22 columns (excluding ID and Output ) with which we are trying to predict if a person smokes or not. 
### Dataset description

The original dataset has 159256 rows. We are Taking 5000 Random Rows from this dataset to be used in streamlit in order to use for visualization. This won't be used for training or test. The file `splitter.py` Has the Logic for this process. 

The dataset which we use for training and test has 154256 rows. The following is the info about the columns. 

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 154256 entries, 0 to 154255
Data columns (total 24 columns):
 #   Column               Non-Null Count   Dtype  
---  ------               --------------   -----  
 0   id                   154256 non-null  int64  
 1   age                  154256 non-null  int64  
 2   height(cm)           154256 non-null  int64  
 3   weight(kg)           154256 non-null  int64  
 4   waist(cm)            154256 non-null  float64
 5   eyesight(left)       154256 non-null  float64
 6   eyesight(right)      154256 non-null  float64
 7   hearing(left)        154256 non-null  int64  
 8   hearing(right)       154256 non-null  int64  
 9   systolic             154256 non-null  int64  
 10  relaxation           154256 non-null  int64  
 11  fasting blood sugar  154256 non-null  int64  
 12  Cholesterol          154256 non-null  int64  
 13  triglyceride         154256 non-null  int64  
 14  HDL                  154256 non-null  int64  
 15  LDL                  154256 non-null  int64  
 16  hemoglobin           154256 non-null  float64
 17  Urine protein        154256 non-null  int64  
 18  serum creatinine     154256 non-null  float64
 19  AST                  154256 non-null  int64  
 20  ALT                  154256 non-null  int64  
 21  Gtp                  154256 non-null  int64  
 22  dental caries        154256 non-null  int64  
 23  smoking              154256 non-null  int64  
dtypes: float64(5), int64(19)
memory usage: 28.2 MB
None
```

`EDA.ipynb` File has the Data analysis and Finding which we used to train the models 

### Models Used

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier
5. Ensemble Model - Random Forest
6. Ensemble Model - XGBoost
#### Evaluation Metrics
| **ML Model Name**       | **Accuracy** | **AUC** | **Precision** | **Recall** | **F1** | **MCC** |
| ----------------------- | ------------ | ------- | ------------- | ---------- | ------ | ------- |
| **Logistic Regression** | 0.7466       | 0.8302  | 0.6994        | 0.7390     | 0.7186 | 0.4890  |
| **Decision Tree**       | 0.6936       | 0.6887  | 0.6506        | 0.6489     | 0.6497 | 0.3774  |
| **K-Nearest Neighbor**  | 0.5660       | 0.5727  | 0.5062        | 0.3711     | 0.4282 | 0.0945  |
| **Naive Bayes**         | 0.7167       | 0.7776  | 0.6387        | 0.8132     | 0.7155 | 0.4542  |
| **Random Forest**       | 0.7730       | 0.8582  | 0.7071        | 0.8222     | 0.7604 | 0.5526  |
| **XGBoost**             | 0.7780       | 0.8627  | 0.7172        | 0.8140     | 0.7626 | 0.5596  |
#### Performance Observations

| **ML Model Name**            | **Observation about model performance**                                                                                                                                     |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression**      | I saw this as my baseline model. Accuracy is better than few other Models like KNN                                                                                          |
| **Decision Tree**            | Yielded moderate results but training time was comparatively faster than most models.                                                                                       |
| **K-Nearest Neighbor**       | Performed the poorest across all metrics, particularly with a very low MCC (0.0945). I personally believe kNN is not really a good option when we have a lot of dimensions. |
| **Naive Bayes**              | This was the Fastest model to train. It had a pretty high  Recall (close to XGB)                                                                                            |
| **Random Forest (Ensemble)** | Outperformed nost models and had the best Recall among all models .                                                                                                         |
| **XGBoost (Ensemble)**       | This had the highest accuracy. This performs better for most Tabular data.                                                                                                  |
