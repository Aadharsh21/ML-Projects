# Machine Learning Projects

Notebooks built in Google Colab with Python, pandas, scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn and Plotly.
Open any notebook in Colab with the badge at its top.

## Rain prediction ([LOGISTICWEATHER.ipynb](LOGISTICWEATHER.ipynb))

Predicts whether it will rain tomorrow, using the Rain in Australia dataset from Kaggle (145,460 daily observations
from 49 locations).

- Logistic regression with mean imputation, min-max scaling and one-hot encoded categories
- Split by time: trained on the years before 2015, validated on 2015, tested on 2016 and 2017
- **84.2% test accuracy** (85.4% on validation)

## Credit card fraud detection ([CREDITFRAUD.ipynb](CREDITFRAUD.ipynb))

Flags fraudulent transactions in the Credit Card Fraud Detection dataset from Kaggle: 284,807 transactions, of which
only 492 (0.17%) are fraud.

- A Keras neural network reached 99.9% validation accuracy but caught only 28% of fraud, because almost every
  transaction is legitimate
- Balanced the data by undersampling legitimate transactions, then trained a logistic regression model
- **Caught 88% of fraud (43 of 49)** on a held-out test set, with 1 false alarm among 50 legitimate transactions

## Insurance cost prediction ([LINEARINSURANCE.ipynb](LINEARINSURANCE.ipynb))

Estimates medical insurance charges for 1,338 customers.

- Explored how BMI and smoking relate to charges with an interactive Plotly chart
- Fit separate linear regression models on age for smokers and non-smokers
- At age 37, the models predict about $31,600 in charges for a smoker versus $7,800 for a non-smoker

## K-means clustering ([clustering.ipynb](clustering.ipynb))

Groups the Iris flower measurements with k-means (k = 3 and k = 6) and plots the clusters with their centers.
