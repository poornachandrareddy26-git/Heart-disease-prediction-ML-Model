#  Heart Disease Prediction using Machine Learning

## Project Overview

Heart Disease Prediction is a machine learning classification project that predicts whether a person is likely to have heart disease based on various medical and clinical attributes.

The project implements an end-to-end machine learning pipeline, starting from data loading and preprocessing, followed by feature selection, handling class imbalance, feature scaling, model training, model evaluation, hyperparameter tuning, model serialization, and deployment as a web application.

The project is intended as an educational and machine-learning demonstration. It should **not be used as a substitute for professional medical diagnosis or clinical decision-making**.

---

## Objective

The main objectives of this project are:

* To analyze a heart disease dataset.
* To preprocess and prepare the data for machine learning.
* To identify relevant features for prediction.
* To handle class imbalance using SMOTE.
* To scale numerical features where required.
* To train and compare multiple classification algorithms.
* To evaluate model performance using appropriate metrics.
* To perform hyperparameter tuning.
* To save the trained model and preprocessing objects.
* To create a prediction application.
* To deploy the machine learning application online.

---

# Dataset

The project uses a heart disease dataset containing:

* **303 records**
* **14 columns**
* **13 input features**
* **1 target variable**

### Dataset Features

| Feature    | Description                           |
| ---------- | ------------------------------------- |
| `age`      | Age of the patient                    |
| `sex`      | Sex of the patient                    |
| `cp`       | Chest pain type                       |
| `trestbps` | Resting blood pressure                |
| `chol`     | Serum cholesterol                     |
| `fbs`      | Fasting blood sugar                   |
| `restecg`  | Resting electrocardiographic results  |
| `thalach`  | Maximum heart rate achieved           |
| `exang`    | Exercise-induced angina               |
| `oldpeak`  | ST depression induced by exercise     |
| `slope`    | Slope of the peak exercise ST segment |
| `ca`       | Number of major vessels               |
| `thal`     | Thalassemia-related measurement       |
| `target`   | Heart disease outcome                 |

### Target Variable

The target variable is `target`.

```text
0 → No heart disease
1 → Heart disease
```

The machine learning model learns patterns from the input features and predicts the target class.

---

# Machine Learning Pipeline

The overall pipeline followed in this project is:

```text
Data Collection
       ↓
Data Loading
       ↓
Data Exploration
       ↓
Data Cleaning
       ↓
Missing Value Analysis
       ↓
Duplicate Analysis
       ↓
Outlier Analysis
       ↓
Variable Transformation
       ↓
Feature Selection
       ↓
Train-Test Split
       ↓
Class Imbalance Handling using SMOTE
       ↓
Feature Scaling
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Hyperparameter Tuning
       ↓
Final Model Selection
       ↓
Model Serialization
       ↓
Prediction
       ↓
Deployment
```

---

# 1. Data Loading

The dataset is loaded using the Pandas library.

Example:

```python
import pandas as pd

df = pd.read_csv("heart disease dataset.csv")
```

Pandas DataFrame provides a convenient structure for analyzing and manipulating the dataset.

---

# 2. Exploratory Data Analysis

Before training the models, the dataset was inspected to understand its structure and characteristics.

Some of the operations used include:

```python
df.head()
df.shape
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
```

### Purpose of these operations

| Operation            | Purpose                                      |
| -------------------- | -------------------------------------------- |
| `head()`             | Displays the first few records               |
| `shape`              | Shows number of rows and columns             |
| `info()`             | Displays data types and non-null information |
| `describe()`         | Provides statistical information             |
| `isnull().sum()`     | Checks for missing values                    |
| `duplicated().sum()` | Checks for duplicate records                 |

---

# 3. Data Cleaning

Data cleaning is performed to identify and handle problems that could negatively affect the machine learning model.

The following areas were considered:

* Missing values
* Duplicate records
* Incorrect data types
* Outliers
* Inconsistent data
* Irrelevant information

Clean and consistent data helps the model learn meaningful patterns.

---

#  4. Missing Value Analysis

Missing values represent observations where a particular feature does not contain a valid value.

Missing values were checked using:

```python
df.isnull().sum()
```

Handling missing values is important because many machine learning algorithms cannot directly process missing values.

Depending on the dataset and problem, missing values can be handled through:

* Removing affected records
* Mean imputation
* Median imputation
* Mode imputation
* Advanced imputation techniques

---

#  5. Duplicate Analysis

Duplicate records were checked using:

```python
df.duplicated().sum()
```

Duplicate observations can cause certain records to have more influence during model training than intended.

---

#  6. Outlier Analysis

An outlier is an observation that is significantly different from the majority of observations.

Outliers ca
