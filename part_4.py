import os
import sqlite3
from re import A

import matplotlib.pyplot as plt
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

# Add total statistics to bin activity level
merged["TotalActiveMinutes"] = (
    merged["VeryActiveMinutes"]
    + merged["FairlyActiveMinutes"]
    + merged["LightlyActiveMinutes"]
)

merged["ActiveFraction"] = merged["TotalActiveMinutes"] / 1440


# Dashboard Functions
# --------------------------------------


def get_individual_summary(df, user_id):
    """Returns a summary of all average stats for a single person"""
    user_data = df[df["Id"] == user_id]

    return {
        "Id": user_id,
        "AvgSteps": user_data["TotalSteps"].mean(),
        "AvgCalories": user_data["Calories"].mean(),
        "AvgSedentaryMinutes": user_data["SedentaryMinutes"].mean(),
        "AvgVeryActiveMinutes": user_data["VeryActiveMinutes"].mean(),
        "AvgMinutesAsleep": user_data["MinutesAsleep"].mean(),
        "AvgHeartRate": user_data["AvgHeartRate"].mean(),
        "AvgBMI": user_data["BMI"].mean(),
        "AvgWeightKg": user_data["WeightKg"].mean(),
        "TotalDaysTracked": len(user_data),
    }


def filter_by_date(df, start_date=None, end_date=None):
    """Filter the merged dataframe to a date range"""
    df["date"] = pd.to_datetime(df["date"])
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    return df


def activity_level_classifier(row):
    """Classify a person as sedentary, lightly, moderately, or very active"""
    if row["VeryActiveMinutes"] >= 30:
        return "Very Active"
    elif row["FairlyActiveMinutes"] >= 30:
        return "Fairly Active"
    elif row["LightlyActiveMinutes"] >= 30:
        return "Lightly Active"
    else:
        return "Sedentary"


merged["ActivityLevel"] = merged.apply(activity_level_classifier, axis=1)


def get_population_summary(df):
    """Summary level stats across all users for the dashboard homepage"""
    return {
        "TotalUsers": df["Id"].nunique(),
        "DateRange": (df["date"].min(), df["date"].max()),
        "AvgDailySteps": df["TotalSteps"].mean(),
        "AvgDailyCalories": df["Calories"].mean(),
        "AvgSleepHours": (df["MinutesAsleep"] / 60).mean(),
        "AvgHeartRate": df["AvgHeartRate"].mean(),
        "ActivityDistribution": df["ActivityLevel"].value_counts(normalize=True),
    }


def get_hourly_patterns(hourly_df, metric="Calories"):
    """Returns average metric value by hour of day across all user"""
    col_map = {
        "Calories": "Calories",
        "Steps": "StepTotal",
        "Intensity": "TotalIntensity",
    }
    hourly_df = hourly_df.copy()
    hourly_df["Hour"] = pd.to_datetime(hourly_df["ActivityHour"]).dt.hour

    return hourly_df.groupby("Hour")[col_map[metric]].mean()


AVAILABLE_STATS = {
    "calories": "Calories",
    "steps": "TotalSteps",
    "sleep": "HoursAsleep",
    "bmi": "BMI",
    "weight": "WeightKg",
    "active_minutes": "TotalActiveMinutes",
    "sedentary_minutes": "SedentaryMinutes",
    "heart_rate": "AvgHeartRate",
    "distance": "TotalDistance",
    "active_fraction": "ActiveFraction",
}


def plot_relationship(df, stat_x, stat_y, user_id=None):
    """Scatter plot of two statistics against eachother."""

    # First validate input stats
    if stat_x not in AVAILABLE_STATS or stat_y not in AVAILABLE_STATS:
        valid = list(AVAILABLE_STATS.keys())
        raise ValueError(f"Invalid stat. Choose from: {valid}")

    col_x = AVAILABLE_STATS[stat_x]
    col_y = AVAILABLE_STATS[stat_y]

    # checks to see if a user id was specified and if not it just uses all user data
    plot_df = df[df["Id"] == user_id].copy() if user_id else df.copy()

    # Drop rows where either col_x or col_y are NaN
    plot_df = plot_df.dropna(subset=[col_x, col_y])

    if plot_df.empty:
        print("No data available for this plot")
        return

    # Correlation statistics
    corr = plot_df[col_x].corr(plot_df[col_y])

    # Plotting
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        plot_df[col_x], plot_df[col_y], alpha=0.5, edgecolors="k", linewidths=0.3
    )

    # Trend line
    m, b = np.polyfit(plot_df[col_x], plot_df[col_y], 1)
    x_line = np.linspace(plot_df[col_x].min(), plot_df[col_x].max(), 100)
    ax.plot(
        x_line,
        m * x_line + b,
        color="red",
        linewidth=1.5,
        linestyle="--",
        label="Trend",
    )

    # Labels
    title = f"{col_x} vs {col_y}"
    if user_id:
        title += f" (User {user_id})"

    ax.set_title(title)
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    ax.legend()
    ax.annotate(
        f"r = {corr:.2f}",
        xy=(0.05, 0.93),
        xycoords="axes fraction",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # --- Test get_individual_summary ---
    test_id = merged["Id"].iloc[0]  # grab first available user
    summary = get_individual_summary(merged, test_id)
    print("\n--- Individual Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # --- Test filter_by_date ---
    filtered = filter_by_date(merged.copy(), "2016-04-01", "2016-04-30")
    print(f"\n--- filter_by_date ---")
    print(f"  Rows in April: {len(filtered)}")

    # --- Test get_population_summary ---
    pop = get_population_summary(merged)
    print("\n--- Population Summary ---")
    for k, v in pop.items():
        print(f"  {k}: {v}")

    # --- Test get_hourly_patterns ---
    print("\n--- Hourly Patterns (Calories) ---")
    print(get_hourly_patterns(hourly_calories, metric="Calories"))

    # --- Test plot_relationship ---
    print("\n--- plot_relationship: sleep vs calories ---")
    plot_relationship(merged, "sleep", "calories")

    print("\n--- plot_relationship: bmi vs steps ---")
    plot_relationship(merged, "bmi", "steps")

    # --- Test invalid input handling ---
    print("\n--- plot_relationship: invalid stat ---")
    try:
        plot_relationship(merged, "invalid_stat", "calories")
    except ValueError as e:
        print(f"  Caught expected error: {e}")
