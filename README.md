# Credit Card Fraud Detection
## Overview

This project focuses on predicting the likelihood of a credit card holder defaulting on their payment in the next month. The model is based on demographic and financial information such as age, education, marital status, bill amounts, and payment history. The goal is to predict whether a customer will default (1) or not (0), based on the available features.

The model uses **Random Forest Classifier** to predict credit defaults. It also includes data exploration, preprocessing, and visualization to understand the patterns and imbalances in the dataset.

## Dataset

The dataset contains information about 30,000 credit card holders and their payment histories, along with their personal details. The dataset is used to predict the target variable `default.payment.next.month` (0 = No default, 1 = Default).

The features in the dataset include:
- **LIMIT_BAL**: Credit limit
- **SEX**: Gender (1 = male, 2 = female)
- **EDUCATION**: Education level (1 = graduate school, 2 = university, 3 = high school, 4 = others, 5 = unknown)
- **MARRIAGE**: Marital status (1 = married, 2 = single, 3 = others)
- **AGE**: Age of the individual
- **PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6**: Payment status for the last 6 months
- **BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6**: Bill amounts for the last 6 months
- **PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6**: Payment amounts for the last 6 months
- **default.payment.next.month**: Target variable (1 = default, 0 = no default)

## Tools and Libraries Used

- **Python Libraries**:
  - pandas: Data manipulation and analysis
  - numpy: Numerical computations
  - matplotlib, seaborn: Data visualization
  - scikit-learn: Machine learning algorithms, data preprocessing, and metrics
  - RandomForestClassifier: For building the predictive model
  - train_test_split: For splitting the dataset into training and testing sets
- **Visualization Libraries**:
  - matplotlib
  - seaborn

## Steps

### 1. Data Preprocessing
- **Dropping Unnecessary Columns**: Removed the `ID` column as it is not useful for prediction.
- **Handling Missing Values**: The dataset doesn't have missing values, as checked by `df.isnull().sum()`.
- **Encoding Categorical Variables**: Categorical variables such as `SEX`, `EDUCATION`, and `MARRIAGE` are represented as numerical values (1, 2, 3, etc.) to be used in the model.
  
### 2. Exploratory Data Analysis (EDA)
- **Category Distribution**: Visualized the distribution of categorical columns such as `SEX`, `EDUCATION`, and `MARRIAGE` to understand class distribution.
- **Imbalance Detection**: The dataset is imbalanced, with fewer instances of defaults (1) compared to non-defaults (0). This can lead to poor model performance.

### 3. Model Building
- **Splitting the Data**: Split the data into features (`X`) and target (`y`), then into training and testing sets (70% train, 30% test).
- **Model**: The Random Forest Classifier was used for classification. The model was trained and evaluated based on precision, recall, and F1-score.

### 4. Model Evaluation
- **Classification Report**: The model achieved an accuracy of 81%, with a recall of 36% for detecting defaults, which is crucial for identifying high-risk customers.
- **Confusion Matrix**: To evaluate the model’s performance by checking the true positives, false positives, and other metrics.

### 5. Results
              precision    recall  f1-score   support

           0       0.84      0.94      0.89      7040
           1       0.62      0.36      0.46      1960

    accuracy                           0.81      9000
   macro avg       0.73      0.65      0.67      9000
weighted avg       0.79      0.81      0.79      9000


### 6. visualization
- **Bar Plots: For visualizing the counts of categorical features (`SEX`, `EDUCATION`, etc.) against the target variable `default.payment.next.month`.
- **Pie Chart: Displayed the imbalance between the two classes (default vs. non-default)

## How to Use This Project

### Clone the Repository:
```bash
git clone https://github.com/yourusername/credit-default-prediction.git
cd credit-default-prediction
```
### Install Required Libraries: 
```bash
pip install -r requirements.txt
```
###Run Model:
```bash
jupyter notebook CreditCardFraudDetection.ipynb
```

## View Results
- **After running the cells in the notebook, it will display the classification report and confusion matrix for evaluation the model's performance.

## Future Improvements
- **Additional Models: Experiment with other machine learning models like Logistic Regression
- **Feature Engineering: Introduce additional features that could improve model performance.

## Conclusion
- **This project demonstrates how a Random Forest Classifier can be used to predict credit defaults based on customer information. By focusing on important features like bill amounts, payment history, and demographic data, I can identify high-risk customers and reduce financial risks for banks and other institutions.





