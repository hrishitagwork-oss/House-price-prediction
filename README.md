# 🏠 House Price Prediction using Machine Learning

## 📌 Overview

This project focuses on predicting house prices using **Machine Learning and Python**.

The project uses a housing dataset containing information about properties such as living area, bedrooms, bathrooms, floors, waterfront availability, property grade, basement area, location, and other housing-related features.

The project follows a complete machine learning workflow, starting from **data loading and exploratory data analysis (EDA)** to **data cleaning, feature selection, model building, prediction, and evaluation**.

Different regression techniques are explored, including:

- Linear Regression
- Multiple Linear Regression
- Polynomial Features
- Ridge Regression

The models are evaluated using the **R² (R-squared) score**.

---

## 🎯 Objective

The main objective of this project is to build regression models that can learn the relationship between different house features and their prices.

The project aims to:

- Understand the housing dataset
- Perform exploratory data analysis
- Handle missing values
- Select relevant features
- Build regression models
- Apply polynomial feature transformation
- Use Ridge Regression for regularization
- Predict house prices
- Evaluate model performance using R² score

---

## 📊 Dataset

The project uses a housing dataset containing **21,613 records**.

The target variable is:

```text
price
````

Some of the important features used in the project include:

| Feature         | Description                          |
| --------------- | ------------------------------------ |
| `bedrooms`      | Number of bedrooms                   |
| `bathrooms`     | Number of bathrooms                  |
| `sqft_living`   | Living area in square feet           |
| `sqft_lot`      | Lot area in square feet              |
| `floors`        | Number of floors                     |
| `waterfront`    | Waterfront property indicator        |
| `view`          | View rating                          |
| `condition`     | Condition of the property            |
| `grade`         | Overall grade of the house           |
| `sqft_above`    | Square footage above ground          |
| `sqft_basement` | Basement square footage              |
| `lat`           | Latitude                             |
| `long`          | Longitude                            |
| `sqft_living15` | Average living area of nearby houses |
| `sqft_lot15`    | Average lot area of nearby houses    |

---

## 🛠️ Tools & Technologies

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

### Machine Learning Techniques

* Linear Regression
* Multiple Linear Regression
* Polynomial Features
* Ridge Regression
* Train-Test Split
* Data Imputation
* Feature Scaling
* Model Evaluation

---

## 🔍 Exploratory Data Analysis

The project begins with exploring the dataset using Pandas.

The following operations are performed:

* Displaying the first few rows
* Checking dataset information
* Checking data types
* Generating descriptive statistics
* Identifying missing values
* Understanding numerical features

Functions such as:

```python
df.head()
```

```python
df.describe()
```

and:

```python
df.dtypes
```

are used for initial data exploration.

---

## 🧹 Data Cleaning

The dataset is cleaned before applying machine learning models.

The project includes:

* Removing unnecessary identifier columns
* Checking for missing values
* Identifying missing values in `bedrooms` and `bathrooms`
* Replacing missing values using the mean
* Preparing the dataset for machine learning

For example:

```python
mean = df['bedrooms'].mean()
df['bedrooms'].replace(np.nan, mean, inplace=True)
```

Similarly, missing values in the `bathrooms` column are handled using the mean value.

---

## 📈 Data Visualization

Visualizations are used to understand relationships between different housing features and house prices.

### House Price vs Waterfront

A box plot is used to examine the distribution of house prices based on waterfront availability.

```python
sns.boxplot(x='waterfront', y='price', data=df)
plt.title('House Price Distribution by Waterfront View')
```

### Square Footage vs Price

A regression plot is used to study the relationship between:

```text
sqft_above
```

and:

```text
price
```

```python
sns.regplot(x='sqft_above', y='price', data=df)
```

These visualizations help in understanding patterns within the housing data.

---

# 🤖 Machine Learning Models

## 1. Simple Linear Regression

The first model uses only:

```text
sqft_living
```

to predict:

```text
price
```

The model is created using:

```python
LinearRegression()
```

The model achieved an R² score of approximately:

```text
0.4929
```

---

## 2. Multiple Linear Regression

The next model uses multiple housing features.

The selected features include:

```text
floors
waterfront
lat
bedrooms
sqft_basement
view
bathrooms
sqft_living15
sqft_above
grade
sqft_living
```

The Multiple Linear Regression model achieved an R² score of approximately:

```text
0.6577
```

---

## 3. Machine Learning Pipeline

A Scikit-learn Pipeline is used to combine multiple preprocessing and modeling steps.

The pipeline includes:

```text
SimpleImputer
      ↓
StandardScaler
      ↓
PolynomialFeatures
      ↓
LinearRegression
```

This approach creates a structured machine learning workflow and achieved an R² score of approximately:

```text
0.7513
```

---

## 4. Ridge Regression

Ridge Regression is applied to reduce the effect of overfitting by introducing regularization.

The dataset is divided into:

```text
80% Training Data
20% Testing Data
```

using:

```python
train_test_split(
    X_imputed_df,
    y,
    test_size=0.2,
    random_state=1
)
```

The Ridge model uses:

```python
Ridge(alpha=0.1)
```

The model achieved an R² score of approximately:

```text
0.6459
```

on the test data.

---

## 5. Polynomial Features + Ridge Regression

The final approach uses **second-degree polynomial features** combined with Ridge Regression.

Polynomial features are generated using:

```python
PolynomialFeatures(degree=2)
```

and Ridge Regression is applied with:

```python
Ridge(alpha=0.1)
```

The model achieved an R² score of approximately:

```text
0.7544
```

on the test data.

---

# 📊 Results

The project produced the following R² scores:

| Model                                  | R² Score |
| -------------------------------------- | -------: |
| Simple Linear Regression               |   0.4929 |
| Multiple Linear Regression             |   0.6577 |
| Pipeline + Polynomial Features         |   0.7513 |
| Ridge Regression                       |   0.6459 |
| Polynomial Features + Ridge Regression |   0.7544 |

> Note: The models were not all evaluated using exactly the same methodology. The later Ridge Regression models use an 80/20 train-test split, while some earlier models were evaluated on the data used for fitting. Therefore, the scores should not be considered a strict apples-to-apples comparison.

---

# 📌 Key Findings

* House price can be modeled using multiple property-related features.
* `sqft_living` provides a useful starting feature for price prediction.
* Using multiple features provides more information than using a single feature.
* Data preprocessing is an important part of the machine learning workflow.
* Polynomial features can help represent more complex relationships.
* Ridge Regression introduces regularization into the model.
* The final Polynomial Features + Ridge Regression approach achieved an R² score of **0.7544 on the test data** in this project.

---

# 🔄 Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Exploration
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Exploratory Data Analysis
   ↓
Feature Selection
   ↓
Linear Regression
   ↓
Multiple Linear Regression
   ↓
Polynomial Features
   ↓
Ridge Regression
   ↓
House Price Prediction
   ↓
Model Evaluation
```

---

# 📁 Project Structure

```text
House-Price-Prediction/
│
├── House_price_prediction.ipynb
├── housing-price-prediction.csv
├── README.md
└── requirements.txt
```

---

# 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/House-Price-Prediction.git
```

### 2. Navigate to the Project Folder

```bash
cd House-Price-Prediction
```

### 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Or install the dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Open the Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
House_price_prediction.ipynb
```

Run the notebook cells sequentially.

---

# 📚 What I Learned

Through this project, I gained practical experience in:

* Python for data analysis
* Pandas and NumPy
* Exploratory Data Analysis
* Data cleaning
* Handling missing values
* Data visualization
* Feature selection
* Linear Regression
* Multiple Linear Regression
* Polynomial Features
* Ridge Regression
* Train-Test Split
* Machine Learning Pipelines
* Model prediction
* Model evaluation using R²

---

# 🔮 Future Improvements

The project can be further improved by:

* Adding MAE, MSE, and RMSE evaluation
* Performing cross-validation
* Applying hyperparameter tuning
* Comparing additional machine learning algorithms
* Performing detailed outlier analysis
* Adding correlation analysis
* Creating actual vs predicted price visualizations
* Building an interactive web application
* Deploying the prediction model
* Allowing users to enter house details and receive predicted prices

---

# 💼 Project Highlights

### Dataset

**21,613 housing records**

### Target Variable

**House Price (`price`)**

### Main Technologies

**Python | Pandas | NumPy | Matplotlib | Seaborn | Scikit-learn**

### Machine Learning

**Linear Regression | Multiple Regression | Polynomial Features | Ridge Regression**

### Evaluation Metric

**R² Score**

### Final Test R²

**0.7544**

---

# 👩‍💻 Author

**Hrishita Gain**

Aspiring Data Analyst and Python enthusiast interested in:

* Data Analytics
* Machine Learning
* Python
* Data Visualization
* Web Development
