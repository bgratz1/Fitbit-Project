import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np

base_directory = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base_directory, "utils", "data", "daily_activity.csv"))

unique_users = df["Id"].nunique()
print(f"{unique_users} unique users")
distance_by_user = df.groupby("Id")["TotalDistance"].sum()
print(distance_by_user)
plt.figure(figsize=(12, 6))
plt.bar(distance_by_user.index.astype(str), distance_by_user.values)
plt.xlabel("User Id")
plt.ylabel("Total Distance")
plt.title("Total Distance by User")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
df["ActivityDate"] = pd.to_datetime(df["ActivityDate"])


def plot_calories_by_user(df, user_id, start_date=None, end_date=None):
    # Filter by user
    user_data = df[df["Id"] == user_id]

    if user_data.empty:
        print(f"No data found for user {user_id}")
        return

    # Filter by date range if provided
    if start_date:
        start_date = pd.to_datetime(start_date)
        user_data = user_data[user_data["ActivityDate"] >= start_date]

    if end_date:
        end_date = pd.to_datetime(end_date)
        user_data = user_data[user_data["ActivityDate"] <= end_date]

    # Sort by date (important for line plots)
    user_data = user_data.sort_values("ActivityDate")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(user_data["ActivityDate"], user_data["Calories"])
    plt.xlabel("Date")
    plt.ylabel("Calories Burned")
    plt.title(f"Daily Calories Burned for User {user_id}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


plot_calories_by_user(df, user_id=8378563200, start_date=None, end_date=None)


# Monday=0, Sunday=6
df["DayOfWeek"] = df["ActivityDate"].dt.day_name()
day_counts = (
    df["DayOfWeek"]
    .value_counts()
    .reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
)
plt.figure(figsize=(8, 5))
plt.bar(day_counts.index, day_counts.values)
plt.xlabel("Day of the Week")
plt.ylabel("Number of Workouts")
plt.title("Workout Frequency by Day of the Week")
plt.show()

df["TotalSteps"] = pd.to_numeric(df["TotalSteps"], errors="coerce")
df["Calories"] = pd.to_numeric(df["Calories"], errors="coerce")

# Drop rows with missing data
df = df.dropna(subset=["TotalSteps", "Calories", "Id"])

# Ensure Id is categorical
df["Id"] = df["Id"].astype("category")

# -----------------------------
# 2. Create design matrix
# -----------------------------
# Dummy variables for Id
X = pd.get_dummies(df[["TotalSteps", "Id"]], drop_first=True)

# Convert all columns to float to avoid dtype issues
X = X.astype(float)

# Add constant for intercept
X = sm.add_constant(X)

# Response variable
y = df["Calories"].astype(float)

# -----------------------------
# 3. Fit OLS regression
# -----------------------------
model = sm.OLS(y, X).fit()
print(model.summary())


# -----------------------------
# 4. Function to plot a user
# -----------------------------
def plot_calories_vs_steps(df, user_id, model):
    user_data = df[df["Id"] == user_id]
    if user_data.empty:
        print(f"No data for user {user_id}")
        return

    # Create dummy variables consistent with model
    X_user = pd.get_dummies(user_data[["TotalSteps", "Id"]], drop_first=True)
    X_user = sm.add_constant(X_user, has_constant="add")
    X_user = X_user.astype(float)  # make sure numeric

    # Ensure all columns match model
    for col in model.params.index:
        if col not in X_user.columns:
            X_user[col] = 0
    X_user = X_user[model.params.index]

    # Predict calories
    user_data["PredictedCalories"] = model.predict(X_user)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(
        user_data["TotalSteps"], user_data["Calories"], label="Actual", color="blue"
    )
    plt.plot(
        user_data["TotalSteps"],
        user_data["PredictedCalories"],
        color="red",
        label="Predicted",
    )
    plt.xlabel("Total Steps")
    plt.ylabel("Calories Burned")
    plt.title(f"Calories vs Steps for User {user_id}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# 5. Example usage
# -----------------------------
plot_calories_vs_steps(df, user_id=8877689391, model=model)


corr_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap = 'coolwarm')
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=60)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# calories burned by activity type

model_df = df[[
    "Calories",
    "VeryActiveDistance",
    "ModeratelyActiveDistance",
    "LightActiveDistance",
]].dropna()

X = model_df[[
    "VeryActiveDistance",
    "ModeratelyActiveDistance",
    "LightActiveDistance",
]]

X = sm.add_constant(X)
y = model_df["Calories"]

model = sm.OLS(y, X).fit()
print(model.summary())

coefficients = model.params[1:]

coefficients.plot(kind='bar')
plt.title("Impact of Distance by Different Activity Levels on Calories")
plt.ylabel("Expected Calories Burned by Additional Km of Activity")
plt.xlabel("Activity Level")
plt.xticks(rotation=10)
plt.show()