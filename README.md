![](UTA-DataScience-Logo.png)

# Network-Attack-Classification Project

This repository presents a project that aims to classify network traffic sessions as either normal or attack using the UNSW-NB15 benchmark dataset. The classification task leverages tabular data and employs various supervised machine learning models. The UNSW-NB 15 dataset was created by the IXIA PerfectStorm tool in the Cyber Range Lab of the Australian Centre and it holds raw network packet features (https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15/data).

## Overview

The goal of this project is to detect network attacks in traffic data using network packet features such as the source of the ip address, destination ip address, transaction protocol, service, source bites, attack type, etc. The task is framed as a binary classification problem and is already encoded with 0 = 'Normal' , 1 = 'Attack'. The pipeline includes data cleaning/preprocessing, model training, feature selection, feature engineering, model testing and performance evaluation on a validation set. The dataset was split into a training/test/validation set and three machine learning models were used for model traing (Logistic Regression, XGBoost, and Random Forest). The models were used as baselines then further enhanced to see if they would improve. The XGBoost model performed the best across all metrics and the model that had engineered features was the overall best performing with the higest F1 score: 0.980 and best confusion matrix. This model was then used to evaluate the validation set and the F1 score produced was 0.9824 indicating a strong balance between precision and recall and it also has a slighly improved confusion matrix. This suggest the XGBoost model with feature engineering is highly effective and robust in its attack detection capabilities. 

## Summary of Work Done



### Data
* **Dataset**: UNSW-NB15 dataset
* **Type**: Tabular CSV file
   * **Input**: CSV format file containing network packet features
   * **Output**: Target variable: label (0 = Normal, 1 = Attack)
* **Size**: The original dataset had 82,332 instances and 45 features.
* The dataset is relatively balanced:

    * Normal: ~37,000 samples

    * Attack: ~45,332 samples (9 types)
* **Instances (Train, Test, Validation Split)**: The data was split into train, validation, and test sets (60/20/20). This occured after every three models.


#### Preprocessing / Clean Up
* Missing values: The dataset contained no missing values.
* Categorical values: All categorical features were converted to numerical values using Label Encoding to be compatible with machine learning algorithms. The reason they were not one-hot encoded was to reduce dimensionality.
* Rescaling: StandardScaler was applied to scale numerical features justified by its robustness to outliers and its ability to preserve the shape of skewed distributions.
* Feature selection: Features with low importance (below a threshold of 0.1) based on feature correlation with the traget variable 'label'.
* Feature engineering: New features were created to potentially capture more complex relationships within the data:
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
* Models: Three different models were evaluated- Logistic Regression, Random Forest, and XGBoost

### Training

* Models were trained using Python3, scikit-learn and other libraries including pandas and numpy.
* Training times varied depending on the model complexity, with XGBoost and Random Forest taking the least amount of time.
* Training curves were not explicitly generated in this analysis.
* No major difficulties were encountered.

### Performance Comparison
* Key performance metric(s): Accuracy, Precision, Recall, F1 Score, and ROC AUC were used to evaluate model performance. The F1 score was chosen as the primary metric for model selection due to its balance between precision and recall.
Show/compare results in one tables

* Bar charts were also generated to compare model performance across the different metrics along with the confusion matrices:
  ![Unknown-16](https://github.com/user-attachments/assets/3f13a41c-6f02-4796-af4d-2309d262c337)



### Conclusions
* Decision tree models showed to work best with this data set as Logistic Regression had the lowest performance consistently across all metrics.
* Feature engineering improved the performance of all models.
* Random Forest and XGBoost had similar performaces with high metric outputs, but XGBoost was slightly better performing.
* The final XGBoost model with feature engineering demonstrates high accuracy and a strong balance between precision and recall(F1 score), making it suitable for network attack detections.

### Future Work
* Explore more advanced feature engineering techniques to further improve model performance.
* Experiment with other machine learning algorithms, such as deep learning models, to see if they can achieve even better results.
Evaluate the model's performance on a larger and more diverse dataset to assess its generalization capabilities.


## How to reproduce results

To reproduce the results of this project, follow these steps:
1. Download the dataset: Download the "UNSW_NB15" dataset from Kaggle (using the provided code).
2. Open the notebook: Open the provided notebook containing the code for data preprocessing, model training, and evaluation.
3. Run the code cells: Execute the code cells in the notebook sequentially to reproduce the results.
**Resources:**
Google Colab: Use Google Colab or Jupyter Notebook to run the code and leverage its computational resources.
Kaggle: Access the dataset and potentially explore other related datasets.

### Overview of files in repository
* NAC-Data-Analysis (1).ipynb: Initial look at the dataset features along with visualization.
* NAC-Data-Cleaning (2).ipynb: Cleaning the dataset, encoding categorical values in preperation for machine learning.
* NAC-Initial-Modeling (3).ipynb: Initial results of dataset after modeling Logistic Regression, XGBoost, and Random Forest before any changes to the data frame.
* NAC-Model-Training-Testing (4).ipynb: Feature selection, and feature engineering with the models retrained after each.
* NAC-Comparison-Evaluation (5).ipynb: Loads all 9 trained models and compares results, and evaluates performance on validation sample.
* Network-Attack-Classification-full (6): shows all the .ipynbs files above put together
 
### Software Setup
* Required Packages: This project uses the following Python packages:
  * Standard Libraries:
   * pandas
   * numpy
   * matplotlib
   * seaborn
   * sklearn (scikit-learn)
* Additional Libraries:
   * kagglehub (For downloading the dataset from Kaggle)


### Data
* The dataset used in this project, "Web Page Phishing Detection Dataset," is available on Kaggle. You can download it directly using the following code:
  
    import kagglehub
  
    path = kagglehub.dataset_download("mrwellsdavid/unsw-nb15")
  
    print("Path to dataset files:", path)


* Preprocessing:
 Run the NAC-Data-Cleaning (2).ipynb notebook which contains code cells related to datarescaling and categorical encoding.

### Training
* To train the models:
  1. Open the NAC-Initial-Modeling (3).ipynb notebook.
  2. Run all cells to see the metrics at the baseline, then open the NAC-Model-Traing-Testing (4).ipynb notebook.
  3. Execute the code in the NAC-Model-Trainingg-Testing (4).ipynb notebook to train and evaluate the three different models after changess to the dataset were made.

#### Performance Evaluation
* To evaluate model performance:
  1. Ensure the models have been trained (by completing the training steps).
  2. Run the notebook NAC-Comparison-Evaluation (5).ipynb . This code is to evaluate the models using various metrics like accuracy,precision, recall, F1 score, and ROC AUC, and display the results. You'll also find code for creating confusion matrices to visualize model performance. It also showd the performance on the validation set and the final summary analysis.


## **Citations**
* David, M. W. (2019, January 29). Unsw_nb15. Kaggle. https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15/data 
