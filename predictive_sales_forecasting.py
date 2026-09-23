import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print("PREDICTIVE SALES FORECASTING")
print("=" * 60)

# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------

print("\nLoading dataset...")

file_name = "sample_-_superstore.xls"

df = pd.read_excel(
    file_name,
    sheet_name="Orders",
    engine="xlrd"
)

print("Dataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# ---------------------------------------------------------
# 2. DATA PREPROCESSING
# ---------------------------------------------------------

print("\nPreprocessing data...")

# Convert Order Date to datetime
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

# Convert Sales to numeric
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")

# Remove missing values
df = df.dropna(subset=["Order Date", "Sales"])

# Remove invalid sales values
df = df[df["Sales"] >= 0]

print("Data preprocessing completed!")
print("Rows after cleaning:", len(df))

# ---------------------------------------------------------
# 3. CREATE MONTHLY SALES DATA
# ---------------------------------------------------------

print("\nCreating monthly sales data...")

monthly_sales = (
    df.set_index("Order Date")
      .resample("MS")["Sales"]
      .sum()
      .reset_index()
)

monthly_sales.columns = ["Date", "Sales"]

print("Number of months:", len(monthly_sales))

# ---------------------------------------------------------
# 4. CREATE TIME FEATURES
# ---------------------------------------------------------

monthly_sales["Year"] = monthly_sales["Date"].dt.year
monthly_sales["Month"] = monthly_sales["Date"].dt.month

# Sequential time index
monthly_sales["Time_Index"] = np.arange(len(monthly_sales))

# ---------------------------------------------------------
# 5. SAVE MONTHLY SALES DATA
# ---------------------------------------------------------

monthly_sales.to_csv(
    "monthly_sales.csv",
    index=False
)

print("Monthly sales data saved as monthly_sales.csv")

# ---------------------------------------------------------
# 6. VISUALIZE HISTORICAL SALES
# ---------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales["Date"],
    monthly_sales["Sales"],
    marker="o"
)

plt.title("Monthly Historical Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "historical_sales.png",
    dpi=300
)

plt.show()

# ---------------------------------------------------------
# 7. TRAIN-TEST SPLIT
# ---------------------------------------------------------

print("\nPreparing training and testing data...")

# Use first 80% for training
train_size = int(len(monthly_sales) * 0.80)

train = monthly_sales.iloc[:train_size]
test = monthly_sales.iloc[train_size:]

X_train = train[["Time_Index"]]
y_train = train["Sales"]

X_test = test[["Time_Index"]]
y_test = test["Sales"]

print("Training observations:", len(train))
print("Testing observations:", len(test))

# ---------------------------------------------------------
# 8. TRAIN LINEAR REGRESSION MODEL
# ---------------------------------------------------------

print("\nTraining Linear Regression model...")

model = LinearRegression()

model.fit(X_train, y_train)

print("Model training completed!")

# ---------------------------------------------------------
# 9. PREDICT TEST DATA
# ---------------------------------------------------------

test["Predicted_Sales"] = model.predict(X_test)

# ---------------------------------------------------------
# 10. MODEL EVALUATION
# ---------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    test["Predicted_Sales"]
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test["Predicted_Sales"]
    )
)

r2 = r2_score(
    y_test,
    test["Predicted_Sales"]
)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# ---------------------------------------------------------
# 11. SAVE PREDICTIONS
# ---------------------------------------------------------

test_output = test[
    [
        "Date",
        "Sales",
        "Predicted_Sales"
    ]
]

test_output.to_csv(
    "sales_predictions.csv",
    index=False
)

print("\nTest predictions saved as sales_predictions.csv")

# ---------------------------------------------------------
# 12. ACTUAL VS PREDICTED GRAPH
# ---------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    test["Date"],
    test["Sales"],
    marker="o",
    label="Actual Sales"
)

plt.plot(
    test["Date"],
    test["Predicted_Sales"],
    marker="o",
    label="Predicted Sales"
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "actual_vs_predicted.png",
    dpi=300
)

plt.show()

# ---------------------------------------------------------
# 13. FUTURE FORECAST
# ---------------------------------------------------------

print("\nGenerating future sales forecast...")

future_months = 12

last_index = monthly_sales["Time_Index"].max()

future_indices = np.arange(
    last_index + 1,
    last_index + future_months + 1
)

future_dates = pd.date_range(
    start=monthly_sales["Date"].max() + pd.DateOffset(months=1),
    periods=future_months,
    freq="MS"
)

future_predictions = model.predict(
    future_indices.reshape(-1, 1)
)

forecast = pd.DataFrame({
    "Date": future_dates,
    "Forecasted_Sales": future_predictions
})

# Prevent negative predictions
forecast["Forecasted_Sales"] = forecast[
    "Forecasted_Sales"
].clip(lower=0)

# ---------------------------------------------------------
# 14. SAVE FUTURE FORECAST
# ---------------------------------------------------------

forecast.to_csv(
    "future_sales_forecast.csv",
    index=False
)

print("Future forecast saved as future_sales_forecast.csv")

# ---------------------------------------------------------
# 15. DISPLAY FUTURE FORECAST
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("NEXT 12 MONTH SALES FORECAST")
print("=" * 60)

print(forecast.to_string(index=False))

# ---------------------------------------------------------
# 16. FUTURE FORECAST GRAPH
# ---------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales["Date"],
    monthly_sales["Sales"],
    label="Historical Sales"
)

plt.plot(
    forecast["Date"],
    forecast["Forecasted_Sales"],
    marker="o",
    label="Future Forecast"
)

plt.title("Historical Sales and Future Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "future_sales_forecast.png",
    dpi=300
)

plt.show()

# ---------------------------------------------------------
# 17. FINAL SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nFiles created:")

print("1. monthly_sales.csv")
print("2. sales_predictions.csv")
print("3. future_sales_forecast.csv")
print("4. historical_sales.png")
print("5. actual_vs_predicted.png")
print("6. future_sales_forecast.png")

print("\nModel:")
print("Linear Regression")

print("\nEvaluation:")
print(f"MAE  = {mae:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"R²   = {r2:.4f}")

print("\nForecast period:")
print("Next 12 months")

print("\nPredictive analytics project completed!")