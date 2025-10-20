# Task5-Data-Science

# Description
This project involves performing Text Classification on the consumer complaint dataset from catalog.data.gov/dataset/consumer-complaint-database. The goal is to classify consumer complaints into four target categories using a variety of machine learning and deep learning models and then comparing their performance.

# Target Categories
| Encoded Label  | Category Name                                     |
|----------------|---------------------------------------------------|
| 0              |Credit reporting or other personal consumer reports| 
| 1              |Debt collection                                    | 
| 2              |Consumer Loan                                      | 
| 3              |Mortgage                                           | 

# Tech Stack 

- Python 
- Environment Google Collab/Jupyter Notebook
- Libraries:
    - Data Handling: pandas, numpy, scipy
    - Text Pre-processing: nltk, TfidfVectorizer
    - Machine Learning: MultinomialNB, LinearSVC, lightgbm
    - Deep Learning: tensorflow.keras, torch
    - Evaluation: sklearn.metrics, matplotlib

# Setup and Execution 

The analysis was performed in a Google Colab environment utilizing a GPU runtime for accelerated training of the Deep Neural Network (DNN). This repository contains: 
- Data Preprocessing File
- Model Comparison (.ipynb)

First run the Data preprocessing code on the original data, the code will output a folder containing tfidf_features.npz, tfidf_vectorizer.pkl, complaints_cleaned2.csv.

The Model comparison has been done on the pre processed data.

Model Training Code Setup in Google Collab 
![Code Setup](./screenshots/Setup.png)

# Model Selection and Training 

| Model Number   | Model Name                     |
|----------------|--------------------------------|
| 0              |Multinomial Naive Bayes         | 
| 1              |Linear Support Vector Classifier| 
| 2              |LightGBM                        | 
| 3              |Simple Deep Neural Network      |

Four distinct multi-class classification models were selected, trained, and evaluated. The input features for all models were the TF-IDF vectors derived from the consumer complaint narratives.

The DNN was a two-layer sequential model, including an input layer, a hidden Dense(128, activation='relu') layer, and an output Dense(4, activation='softmax') layer. It was trained for 5 epochs using a memory-efficient SparseDataGenerator to handle the large, sparse TF-IDF input matrix.

Model Training
![Model Training](./screenshots/model_training.png)


# Model Comparison 

| Test Accuracy  | Model Name                     |
|----------------|--------------------------------|
| 86.38%         |Multinomial Naive Bayes         | 
| 91.46%         |Linear Support Vector Classifier| 
| 91.16%         |LightGBM                        | 
| 93.59%         |Simple Deep Neural Network      | 

Model Evaluation Output
![Model Evaluation](./screenshots/model_eval.png)

Model Comparison
![Comparison](./screenshots/model_compare.png)

The DNN achieved the highest overall test accuracy and showed strong, balanced performance across the major categories.



| Category          | Precision      | Recall   | F1 Score  |
|-------------------|----------------|----------|-----------|
| Credit reporting  |   0.95         |  0.97    |   0.96    |
| Debt collection   |   0.92         |  0.88    |   0.90    |
| Consumer Loan     |   0.63         |  0.55    |   0.59    |
| Mortgage          |   0.94         |  0.94    |   0.94    |

The confusion matrix confirms high recall for the Credit reporting and Mortgage categories, while the lower support Consumer Loan category is the most challenging for the model to classify, resulting in the lowest F1-score.

Confusion matrix
![DNN Confusion Matrix](./screenshots/confusion_m_dnn.png)


# Final Sample Predictions 

Prediction Using Best Models
![Predictions](./screenshots/sample_predictions.png)


The top two models LinearSVC and the DNN were used to predict for sample data. 

Sample Complaint 1: I have been trying to dispute items on my credit report for months. The credit bureaus are not responding to my disputes.

Both models successfully predicted -> Credit reporting, credit repair services, or other personal consumer reports

Sample Complaint 2: A collection agency keeps calling me about a debt that I already paid off. They are harassing me at work.

Both models successfully predicted -> Debt collection


# Conclusion
The Simple Deep Neural Network (DNN) achieved the highest accuracy of 0.9359, making it the most performant model for this multi-class consumer complaint text classification task. 