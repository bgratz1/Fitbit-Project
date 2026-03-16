import os
import sqlite3

import numpy as np
import pandas as pd

base_directory = os.path.dirname(os.path.abspath(__file__))

database = os.path.join(base_directory, "utils", "data", "fitbit_database.db")
daily_activity = pd.read_csv(
    os.path.join(base_directory, "utils", "data", "daily_activity.csv")
)


# Extracting database tables into individual pandas DataFrames
# --------------------------------------
connection = sqlite3.connect(database)
cursor = connection.execute("SELECT * FROM weight_log")

rows = cursor.fetchall()
weight_log = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])

cursor = connection.execute("SELECT * FROM hourly_calories")
rows = cursor.fetchall()
hourly_calories = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])

cursor = connection.execute("SELECT * FROM hourly_intensity")
rows = cursor.fetchall()
hourly_intensity = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])

cursor = connection.execute("SELECT * FROM hourly_steps")
rows = cursor.fetchall()
hourly_steps = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])

cursor = connection.execute("SELECT * FROM minute_sleep")
rows = cursor.fetchall()
minute_sleep = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])

cursor = connection.execute("SELECT * FROM heart_rate")
rows = cursor.fetchall()
heart_rate = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])


# Database Cleaning
# --------------------------------------

# Fill missing weight values by using the median weight
weight_log["WeightKg"] = weight_log["WeightKg"].fillna(weight_log["WeightKg"].median())
weight_log.drop(
    columns=["WeightPounds"], inplace=True
)  # can easily convert Kg to Pounds so dont need to take up the extra space

# Dropping the 'Fat' column since over 90% of the data is missing
weight_log.drop(columns=["Fat"], inplace=True)

# Standardizing date
daily_activity["date"] = pd.to_datetime(daily_activity["ActivityDate"]).dt.date
weight_log["date"] = pd.to_datetime(weight_log["Date"]).dt.date

hourly_calories["date"] = pd.to_datetime(hourly_calories["ActivityHour"]).dt.date
hourly_intensity["date"] = pd.to_datetime(hourly_intensity["ActivityHour"]).dt.date
hourly_steps["date"] = pd.to_datetime(hourly_steps["ActivityHour"]).dt.date
heart_rate["date"] = pd.to_datetime(heart_rate["Time"]).dt.date
minute_sleep["date"] = pd.to_datetime(minute_sleep["date"]).dt.date

# Aggregating the sub daily tables to daily values
daily_calories = hourly_calories.groupby(["Id", "date"])["Calories"].sum().reset_index()
daily_intensity = (
    hourly_intensity.groupby(["Id", "date"])["TotalIntensity"].sum().reset_index()
)
daily_steps_hourly = (
    hourly_steps.groupby(["Id", "date"])["StepTotal"].sum().reset_index()
)

daily_sleep = minute_sleep.groupby(["Id", "date"])["value"].count().reset_index()
daily_sleep.rename(columns={"value": "MinutesAsleep"}, inplace=True)

daily_hr = heart_rate.groupby(["Id", "date"])["Value"].mean().reset_index()
daily_hr.rename(columns={"Value": "AvgHeartRate"}, inplace=True)


# Merging into daily_activity
merged = daily_activity.copy()

merged = merged.merge(
    weight_log[["Id", "date", "WeightKg", "BMI"]], on=["Id", "date"], how="left"
)
merged = merged.merge(daily_calories, on=["Id", "date"], how="left")
merged = merged.merge(daily_intensity, on=["Id", "date"], how="left")
merged = merged.merge(daily_steps_hourly, on=["Id", "date"], how="left")
merged = merged.merge(daily_sleep, on=["Id", "date"], how="left")
merged = merged.merge(daily_hr, on=["Id", "date"], how="left")

merged.drop(columns=["Calories_y"], inplace=True)
merged.rename(columns={"Calories_x": "Calories"}, inplace=True)

# Drop sparse rows with missing crucial data
merged = merged[merged["Calories"] > 0]
merged.drop(
    columns=["LoggedActivitiesDistance", "SedentaryActiveDistance"], inplace=True
)

merged["WeightKg"] = merged.groupby("Id")["WeightKg"].ffill().bfill()
merged["BMI"] = merged.groupby("Id")["BMI"].ffill().bfill()

# Add column to bin sleep quality
merged["HoursAsleep"] = merged["MinutesAsleep"] / 60
merged["SleepCategory"] = pd.cut(
    merged["HoursAsleep"],
    bins=[0, 6, 9, 24],
    labels=["Under-slept", "Normal", "Over-slept"],
)
print(merged.head(50))
