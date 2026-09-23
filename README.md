# Predictive Sales Forecasting Using Historical Data

## Project Overview

This project uses historical sales data to build a predictive model for forecasting future sales trends.

The project was developed as part of a Data Analytics internship and demonstrates the complete predictive analytics workflow:

- Data collection
- Data cleaning and preprocessing
- Historical sales analysis
- Feature engineering
- Linear Regression modeling
- Model evaluation
- Future sales forecasting
- Data visualization

## Objective

The main objective of this project is to analyze historical sales data and use a machine learning regression model to predict future sales.

## Dataset

The project uses the **Superstore Sales Dataset**.

The dataset contains information such as:

- Order Date
- Sales
- Customer information
- Product information
- Category
- Region
- Quantity
- Profit

The analysis focuses mainly on **Order Date** and **Sales**.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Linear Regression
- Excel Dataset

## Project Workflow

### 1. Data Loading

The Superstore dataset is loaded using Pandas.

### 2. Data Preprocessing

The following preprocessing steps were performed:

- Converted Order Date into datetime format
- Converted Sales into numeric values
- Removed missing values
- Removed invalid negative sales values

### 3. Monthly Sales Aggregation

Daily/order-level sales data was converted into monthly sales totals.

This monthly dataset was used for trend analysis and predictive modeling.

### 4. Feature Engineering

A sequential **Time Index** was created to represent the progression of time.

The following features were created:

- Year
- Month
- Time Index

### 5. Model Training

A **Linear Regression** model was used to learn the relationship between time and sales.

The historical data was divided into:

- 80% Training Data
- 20% Testing Data

### 6. Model Evaluation

The model was evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### Model Results

| Metric | Result |
|---|---:|
| MAE | 18858.47 |
| RMSE | 23943.35 |
| R² Score | 0.0038 |

The R² value is low, indicating that a simple linear time trend explains only a small portion of the variation in monthly sales. This provides an opportunity for future improvement using more advanced time-series models and additional features.

## Future Sales Forecast

The trained Linear Regression model was used to generate a forecast for the **next 12 months**.

The forecasted values are stored in:

`future_sales_forecast.csv`

## Visualizations

The project generates the following visualizations:

### Historical Sales

`historical_sales.png`

Shows the monthly historical sales trend.

### Actual vs Predicted Sales

`actual_vs_predicted.png`

Compares actual sales values with the model's predicted values for the testing period.

### Future Sales Forecast

`future_sales_forecast.png`

Shows historical sales together with the forecast for the next 12 months.

## Output Files

```text
Predictive-Sales-Forecasting/
│
├── predictive_sales_forecasting.py
├── sample_-_superstore.xls
├── monthly_sales.csv
├── sales_predictions.csv
├── future_sales_forecast.csv
├── historical_sales.png
├── actual_vs_predicted.png
└── future_sales_forecast.png

Key Learning Outcomes

This project helped demonstrate:

Historical data preprocessing
Time-based feature engineering
Regression modeling
Train-test splitting
Model evaluation
Sales trend analysis
Future forecasting
Data visualization using Python
Exporting analytical results to CSV files
Future Improvements

The forecasting model can be improved by using:

Time Series models such as ARIMA or SARIMA
Prophet
Random Forest Regression
Gradient Boosting
XGBoost
Seasonal features
Lag features
Rolling averages
Additional business features such as Category, Region, and Quantity
Conclusion

This project demonstrates a complete predictive analytics workflow using historical sales data. Linear Regression was used as a baseline forecasting model, evaluated using standard regression metrics, and applied to forecast sales for the next 12 months.
