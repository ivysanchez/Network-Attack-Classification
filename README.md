![](UTA-DataScience-Logo.png)

# Network-Attack-Classification Project

This repository presents a project that aims to classify network traffic sessions as either normal or attack using the UNSW-NB15 benchmark dataset. The classification task leverages tabular data and employs various supervised machine learning models.

## Overview

The goal of this project is to detect network attacks in traffic data using network packet features such as the source of the ip address, destination ip address, transaction protocol, service, source bites, attack type, etc. The task is framed as a binary classification problem and is already encoded with 0 = 'Normal' , 1 = 'Attack'. The pipeline includes data cleaning/preprocessing, model training, feature selection, feature engineering, model testing and performance evaluation on a validation set. The dataset was split into a training/test/validation set and three machine learning models were used for model traing (Logistic Regression, XGBoost, and Random Forest). The models were used as baselines then further enhanced to see if they would improve. The XGBoost model performed the best across all metrics and the model that had engineered features was the overall best performing with the higest F1 score: 0.980 and best confusion matrix. This model was then used to evaluate the validation set and the F1 score produced was 0.9824 indicating a strong balance between precision and recall and it also has a slighly improved confusion matrix. This suggest the XGBoost model with feature engineering is highly effective and robust in its attack detection capabilities. 

## Summary of Work Done


### Data
* **Dataset**: UNSW-NB15 dataset
* **Type**: Tabular CSV file
   * **Input**: CSV format file containing network packet features
   * **Output**: Target variable: label
0 = Normal
1 = Attack
* **Size**: The original dataset had 82,332 instances and 45 features. The dataset is relatively balanced:

Normal: ~37,000 samples
Attack: ~45,332 samples
* **Instances (Train, Test, Validation Split)**: The data was split into an 80% training set and a 20% testing set, with no separate validation set used in this initial exploration.

* fixxxxx

#### Preprocessing / Clean Up
* Missing values: The dataset contained no missing values.
* Categorical values: All categorical features were converted to numerical values using Label Encoding to be compatible with machine learning algorithms. The reason they were not one-hot encoded was to reduce dimensionality.
* Rescaling: StandardScaler was applied to scale numerical features justified by its robustness to outliers and its ability to preserve the shape of skewed distributions.
* Feature selection: Features with low importance (below a threshold of 0.1) based on feature correlation with the traget variable 'label'.
* Feature engineering: New features were created to potentially capture more complex relationships within the data:
#### Data Visualization
* Histograms and count plots were used to visualize the distribution of each feature, providing insights into the data characteristics.
  


* A bar chart was generated to display the top 13 most important features identified by Random Forest feature 

* Bar charts were created to compare the performance of different models across key metrics.
  
  Example of bar chart for F1 score:
  
 * fixxxxxxxxxx


### Problem Formulation
* Input: A set of features extracted from network packets.
* Output: A binary classification (0 for normal, 1 for attack).
* Models: Three different models were evaluated:
Logistic Regression
Random Forest
XGBoost

### Training

* Models were trained using Python3, scikit-learn and other libraries including pandas and numpy.
* Training times varied depending on the model complexity, with XGBoost and Random Forest taking the least amount of time.
* Training curves were not explicitly generated in this analysis.
* No major difficulties were encountered.

### Performance Comparison
* Key performance metric(s): Accuracy, Precision, Recall, F1 Score, and ROC AUC were used to evaluate model performance. The F1 score was chosen as the primary metric for model selection due to its balance between precision and recall.
Show/compare results in one tables

* Bar charts were also generated to compare model performance across the different metrics.

  * fixxxxxxx

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


  * fixxxxxxx
* Preprocessing:
The preprocessing steps are already included in the phishingdetection.ipynb notebook. Run the code cells related to data cleaning, feature scaling, feature selection, and feature engineering before training the models.

### Training
* To train the models:
  1. Open the phishingdetection.ipynb notebook.
  2. Run all cells up to the "Iterative Modeling" section.
  3. Execute the code in the "Iterative Modeling" and "Model Parameter Tuning" sections to train and evaluate different models and optimize hyperparameters.

#### Performance Evaluation
* To evaluate model performance:
  1. Ensure the models have been trained (by completing the training steps).
  2. Run the code cells in the "Iterative Modeling" and "Model Parameter Tuning" sections.These sections contain code to evaluate the models using various metrics like accuracy,precision, recall, F1 score, and ROC AUC, and display the results. You'll also find code for creating confusion matrices to visualize model performance.


## **Citations**
* David, M. W. (2019, January 29). Unsw_nb15. Kaggle. https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15/data 
