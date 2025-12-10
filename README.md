# Insurance Buyer Prediction from Age and Income

A Machine Learning project that predicts whether a customer will purchase insurance based on two key demographic factors — Age and Annual Income. This project uses a Logistic Regression classification model, trained and validated using structured insurance buyer data.

<!-- Title Badges --> <p align="center"> <img src="https://img.shields.io/badge/Model-Logistic%20Regression-blue" /> <img src="https://img.shields.io/badge/Language-Python-green" /> <img src="https://img.shields.io/badge/Notebook-Jupyter-orange" /> <img src="https://img.shields.io/badge/Status-Active-success" /> </p>

## Overview
This repository contains everything required to train, evaluate, and use a machine-learning model that predicts the probability of a customer buying insurance.
The system takes Age and Income as input and outputs the likelihood of purchase (0 = No, 1 = Yes).

This model is helpful for:

* Insurance companies

* Customer segmentation

* Marketing targeting

* Risk assessment

## Project Structure
```
 Insurance Buyer Prediction from Age and Income
│
├── logistic_regression_pattern.ipynb     
├── insurance_model_age.pkl               
├── data/                                 
│   └── <dataset included inside .ipynb>
├── README.md                             
```

## Dataset Description

The dataset (available inside the notebook) contains the following features:

| Column Name |  Description                               |
| ----------  |  ------------------------------------------|
| Age         |  Customer’s age in years                   |
| Income      |  Annual income of customer                 |  
|Insurance    |  Target → 1 (Purchased), 0 (Not Purchased) |

The dataset was cleaned, encoded, and scaled inside the notebook.

## Model Used
Logistic Regression

* Suitable for binary classification

* Interpretable and fast

* Works well with linearly separable features

Why Logistic Regression?

* Age and income have linear relationships with purchase probability

* Output is probabilistic (helps in business decisions)

## How the Model Works
Input:
```
{
  "age": 35,
  "income": 65000
}
```
Output:
```
Prediction: 1
Confidence: 78%
```
The model calculates:

* Weighted combination of features

* Applies sigmoid function

* Predicts class (0/1)

##  How to Run Locally

1. Clone the repository
```
git clone https://github.com/<your-username>/Insurance-Buyer-Prediction.git
cd Insurance-Buyer-Prediction
```

2. Create a virtual environment
```
python -m venv venv
source venv/bin/activate      
venv\Scripts\activate         
```
3. Install dependencies
```
pip install -r requirements.txt
```
4. Open the Jupyter Notebook
```
jupyter notebook logistic_regression_pattern.ipynb
```
5. Load the .pkl model

Use this inside Python:
```
import pickle

with open("insurance_model_age.pkl", "rb") as file:
    model = pickle.load(file)

model.predict([[age, income]])
```
## Model Performance
| Metric               | Score           |
| -------------------- | --------------- |
| **Accuracy**         | ~98%            |
| **Precision**        | High            |
| **Recall**           | High            |
| **Confusion Matrix** | Good separation |

## Notebook Highlights

Inside logistic_regression_pattern.ipynb, you will find:

* Data loading

* Exploratory Data Analysis

* Visualization (Scatter plots, histograms)

* Model training

* Accuracy evaluation

* Final model export to .pkl file

## Future Enhancements

* Add gender, region, marital status features

* Build a Flask/FastAPI web app

* Deploy model using Streamlit

* Build an API endpoint for real-time predictions

* Add SHAP explainability

* Convert the model to ONNX for lightweight inference

## Contributing

Contributions are welcome!

If you want to improve the model or add visual dashboards, open a pull request.

## License

This project is open-source under the MIT License.

## Support

If you like this project, please star ⭐ the repository —> **it motivates further development!**





