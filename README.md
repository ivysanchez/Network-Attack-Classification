![](UTA-DataScience-Logo.png)

# Network Attack Classification Project

This project classifies network traffic sessions as either normal or attack using the UNSW-NB15 benchmark dataset. The dataset, created by the IXIA PerfectStorm tool in the Cyber Range Lab of the Australian Centre, contains raw network packet features (https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15/data).

## Overview

The goal of this project is to detect network attacks in traffic data using network packet features such as the source IP address, destination IP address, transaction protocol, service, source bytes, attack type, and more. The task is framed as a binary classification problem, with labels encoded as 0 = 'Normal' , 1 = 'Attack'. 

The pipeline includes data cleaning/preprocessing, model training, feature selection, feature engineering, model testing, and performance evaluation on a validation set. The dataset was split into training, test, and validation sets, and three machine learning models were trained: Logistic Regression, XGBoost, and Random Forest. These models were first used as baselines and then enhanced to assess performance improvements. 

XGBoost performed the best across all metrics. The model with engineered features achieved the highest F1 score (0.980) and the best confusion matrix on the test set. When evaluated on the validation set, this model achieved an F1 score of 0.9824, indicating a strong balance between precision and recall, along with a slightly improved confusion matrix. These results suggest that the XGBoost model with feature engineering is highly effective and robust for attack detection. 

## Summary of Work Done



### Data
* **Dataset**: UNSW-NB15 dataset
* **Type**: Tabular CSV file
   * **Input**: Network packet features (CSV format)
   * **Output**: Target variable: label (0 = Normal, 1 = Attack)
* **Size**: 82,332 instances, 45 features
* The dataset is relatively balanced:

    * Normal: ~37,000 samples

    * Attack: ~45,332 samples (across9 types)
* **Instances (Train, Test, Validation Split)**: 60/20/20 (applied after training each set of three models)


#### Preprocessing / Clean Up
* Missing values: None found.
* Categorical values: Converted to numerical values using Label Encoding to reduce dimensionality (instead of one-hot encoding).
* Rescaling: Applied StandardScaler to numerical features, chosen for its robustness to outliers and ability to preserve skewed distributions.
* Feature selection: Removed features with low importance (correlation threshold < 0.1) relative to the target variable label.
* Feature engineering: Created new features to capture more complex relationships within the data.
#### Data Visualization
* Histograms and count plots were used to visualize the distribution of each feature, providing insights into the data characteristics.
<img width="711" alt="Screenshot 2025-07-07 at 4 59 10 AM" src="https://github.com/user-attachments/assets/ba5be27b-6c66-493e-a552-15e6d07c96e7" />  


![Unknown](https://github.com/user-attachments/assets/4070ec0b-8bfd-496c-95a4-805e68721449)

* A plot was used to display feature importance through correlation with the target variable:
  
  ![Unknown-11](https://github.com/user-attachments/assets/b9aa9444-47c5-4549-9326-4e0ff75f18ac)


* Bar charts were created to compare the performance of different models across key metrics.
  
  Example of bar chart for F1 score:

![Unknown-15](https://github.com/user-attachments/assets/330a79f7-2f14-4a91-a7e5-14e86b2f61b1)



### Problem Formulation
* Input: A set of features extracted from network packets.
* Output: A binary classification (0 for normal, 1 for attack).
* Models: Three different models were evaluated — Logistic Regression, Random Forest, and XGBoost.

### Training

* Models were trained using Python 3, scikit-learn, and other libraries including pandas and numpy.
* XGBoost and Random Forest had the shortest training times; Logistic Regression was slower on this dataset
* Training curves were not generated
* No major difficulties were encountered.

### Performance Comparison
* Metrics Used: Accuracy, Precision, Recall, F1 Score, ROC AUC
* Primary metric: F1 Score (for its balance between precision and recall)
  
Results were summarized in a comparison table below:

Model Performance Comparison (Original, Selected, and Engineered Features)

| Model             | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------------------|----------|-----------|--------|----------|---------|
| Logistic Regression Baseline Model | 0.890205 | 0.891405 | 0.911658 | 0.901418 | 0.887788 |
| XGBoost Baseline Model | 0.976924 | 0.985145 | 0.972758 | 0.978912 | 0.977393 |
| Random Forest Baseline Model | 0.976498 | 0.983404 | 0.973751 | 0.978554 | 0.976808 |
| Logistic Regression (Selected Features) | 0.875752 | 0.904296 | 0.865998 | 0.884732 | 0.876850 |
| XGBoost (Selected Features) | 0.968057 | 0.972244 | 0.969670 | 0.970955 | 0.967876 |
| Random Forest (Selected Features) | 0.967511 | 0.970238 | 0.970773 | 0.970506 | 0.967143 |
| Logistic Regression (Engineered Features) | 0.888869 | 0.889295 | 0.911658 | 0.900338 | 0.886302 |
| XGBoost (Engineered Features) | 0.978381 | 0.986051 | 0.974523 | 0.980253 | 0.978816 |
| Random Forest (Engineered Features) | 0.975162 | 0.983363 | 0.971325 | 0.977307 | 0.975595 |


* Confusion matrices were also generated for comparison:

![Unknown-16](https://github.com/user-attachments/assets/04828431-d456-4b84-925c-f4257cc4ad34)




### Conclusions
* Decision tree–based models worked best with this dataset, while Logistic Regression consistently had the lowest performance across all metrics.
* Feature engineering improved the performance of all models.
* Random Forest and XGBoost had similar high performance, but XGBoost was slightly better overall.
* XGBoost achieved the highest performance, with:
        * Test set: F1 = 0.980
        * Validation set: F1 = 0.9824, strong precision-recall balance, slightly improved confusion matrix
* Final model: XGBoost with feature engineering was the most suitable for robust network attack detection. See final results on validation set below.

XGBoost Model with Engineered Features 

| Metric        | Value   |
|---------------|---------|
| Accuracy      | 0.9807  |
| Precision     | 0.9869  |
| Recall        | 0.9781  |
| F1 Score      | 0.9824  |
| ROC AUC Score | 0.9811  |

Confusion Matrix

|               | Predicted Normal | Predicted Attack |
|---------------|------------------|------------------|
| **Actual Normal** | 7282             | 118              |
| **Actual Attack** | 199              | 8868             |



### Future Work
* Explore more advanced feature engineering techniques to further improve model performance.
* Experiment with other machine learning algorithms, such as deep learning models, to see if they can achieve better results.
* Evaluate model performance on a larger and more diverse dataset to assess generalization capabilities.



## How to reproduce results

To reproduce the results of this project, follow these steps:
1. Download the dataset
Download the UNSW_NB15 dataset from Kaggle (using the provided code).
Open the notebook
Open the provided notebook containing the code for data preprocessing, model training, and evaluation.
Run the code cells
Execute the code cells sequentially to reproduce the results.
**Resources:**
* Google Colab / Jupyter Notebook: Run the code and leverage computational resources.
* Kaggle: Access the dataset and explore related datasets.

### Overview of Files in Repository

| File Name | Description |
|-----------|-------------|
| **NAC-Data-Analysis (1).ipynb** | Initial dataset exploration and visualizations |
| **NAC-Data-Cleaning (2).ipynb** | Data cleaning, label encoding, and preparation for ML |
| **NAC-Initial-Modeling (3).ipynb** | Baseline performance of Logistic Regression, XGBoost, and Random Forest |
| **NAC-Model-Training-Testing (4).ipynb** | Feature selection and engineering; retraining models |
| **NAC-Comparison-Evaluation (5).ipynb** | Comparison of 9 trained models and validation set evaluation |
| **Network-Attack-Classification-full (6).ipynb** | Consolidated notebook containing all steps above |

### Software Setup
* Standard Libraries:
   * pandas
   * numpy
   * matplotlib
   * seaborn
   * sklearn (scikit-learn)
* Additional Libraries:
   * kagglehub (For downloading the dataset from Kaggle)


### Data
* The dataset used in this project, "UNSW_NB15" is available on Kaggle. You can download it directly using the following code:
  
    import kagglehub
  
    path = kagglehub.dataset_download("mrwellsdavid/unsw-nb15")
  
    print("Path to dataset files:", path)


* Preprocessing:
Run NAC-Data-Cleaning (2).ipynb, which contains code cells for data rescaling and categorical encoding.

### Training
* To train the models:
  1. Open NAC-Initial-Modeling (3).ipynb and run all cells to see baseline metrics.
  2. Open NAC-Model-Training-Testing (4).ipynb and run all cells to train and evaluate the models after dataset modifications.

#### Performance Evaluation
* To evaluate model performance:
  1. Ensure models have been trained.
  2. Run NAC-Comparison-Evaluation (5).ipynb to evaluate models using accuracy, precision, recall, F1 score, and ROC AUC.
  3. This notebook also generates confusion matrices to visualize performance, shows validation set results, and provides the final summary analysis.


## **Citations**
* David, M. W. (2019, January 29). Unsw_nb15. Kaggle. https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15/data 
