import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# --------------------------
# Sample Data
# --------------------------
data = {
    "Revenue": [59000,60500,61800,63000,64500,66000,67200,68800],
    "EBIT_Margin": [24.2,24.5,24.8,25.0,25.1,25.3,25.5,25.8],
    "PAT": [11800,12050,12320,12650,12980,13250,13520,13900],
    "Shares": [362,362,362,362,362,362,362,362],
    "EPS": [32.6,33.3,34.0,34.9,35.8,36.6,37.3,38.4]
}

df = pd.DataFrame(data)

# --------------------------
# Features and Target
# --------------------------
X = df[["Revenue","EBIT_Margin","PAT","Shares"]]
y = df["EPS"]

# --------------------------
# Train/Test Split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# --------------------------
# XGBoost Model
# --------------------------
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------
# Predictions
# --------------------------
predictions = model.predict(X_test)

print("Predicted EPS:")
print(predictions)

print("\nActual EPS:")
print(y_test.values)

print("\nMean Absolute Error:")
print(mean_absolute_error(y_test, predictions))

# --------------------------
# Predict Future Quarter
# --------------------------
future = pd.DataFrame({
    "Revenue":[70000],
    "EBIT_Margin":[26.0],
    "PAT":[14250],
    "Shares":[362]
})

future_eps = model.predict(future)

print("\nPredicted Future EPS:", future_eps[0])