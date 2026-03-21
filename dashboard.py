import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fitbit Dashboard", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_existing_path(*candidates):
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find any of: {candidates}")


CSV_PATH = find_existing_path(
    os.path.join(BASE_DIR, "utils", "data", "daily_activity.csv"),
    os.path.join(BASE_DIR, "daily_activity.csv"),
)

DB_PATH = find_existing_path(
    os.path.join(BASE_DIR, "utils", "data", "fitbit_database.db"),
    os.path.join(BASE_DIR, "fitbit_database.db"),
)


@st.cache_data
def load_activity_csv(path: str) -> pd.DataFrame:
    activity = pd.read_csv(path)
    activity["ActivityDate"] = pd.to_datetime(activity["ActivityDate"])
    return activity


@st.cache_data
def load_merged_data(csv_path: str, db_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_activity = pd.read_csv(csv_path)
    daily_activity["date"] = pd.to_datetime(daily_activity["ActivityDate"]).dt.date
    daily_activity["ActivityDate"] = pd.to_datetime(daily_activity["ActivityDate"])

    conn = sqlite3.connect(db_path)
    weight_log = pd.read_sql_query("SELECT * FROM weight_log", conn)
    hourly_calories = pd.read_sql_query("SELECT * FROM hourly_calories", conn)
    hourly_intensity = pd.read_sql_query("SELECT * FROM hourly_intensity", conn)
    hourly_steps = pd.read_sql_query("SELECT * FROM hourly_steps", conn)
    minute_sleep = pd.read_sql_query("SELECT * FROM minute_sleep", conn)
    heart_rate = pd.read_sql_query("SELECT * FROM heart_rate", conn)
    conn.close()

    weight_log["WeightKg"] = weight_log["WeightKg"].fillna(weight_log["WeightKg"].median())
    for col in ["WeightPounds", "Fat"]:
        if col in weight_log.columns:
            weight_log = weight_log.drop(columns=col)

    weight_log["date"] = pd.to_datetime(weight_log["Date"]).dt.date
    hourly_calories["date"] = pd.to_datetime(hourly_calories["ActivityHour"]).dt.date
    hourly_intensity["date"] = pd.to_datetime(hourly_intensity["ActivityHour"]).dt.date
    hourly_steps["date"] = pd.to_datetime(hourly_steps["ActivityHour"]).dt.date
    heart_rate["date"] = pd.to_datetime(heart_rate["Time"]).dt.date
    minute_sleep["date"] = pd.to_datetime(minute_sleep["date"]).dt.date

    daily_calories = hourly_calories.groupby(["Id", "date"])["Calories"].sum().reset_index()
    daily_intensity = hourly_intensity.groupby(["Id", "date"])["TotalIntensity"].sum().reset_index()
    daily_steps_hourly = hourly_steps.groupby(["Id", "date"])["StepTotal"].sum().reset_index()
    daily_sleep = minute_sleep.groupby(["Id", "date"])["value"].count().reset_index()
    daily_sleep = daily_sleep.rename(columns={"value": "MinutesAsleep"})
    daily_hr = heart_rate.groupby(["Id", "date"])["Value"].mean().reset_index()
    daily_hr = daily_hr.rename(columns={"Value": "AvgHeartRate"})

    merged = daily_activity.copy()
    merged = merged.merge(weight_log[["Id", "date", "WeightKg", "BMI"]], on=["Id", "date"], how="left")
    merged = merged.merge(daily_calories, on=["Id", "date"], how="left")
    merged = merged.merge(daily_intensity, on=["Id", "date"], how="left")
    merged = merged.merge(daily_steps_hourly, on=["Id", "date"], how="left")
    merged = merged.merge(daily_sleep, on=["Id", "date"], how="left")
    merged = merged.merge(daily_hr, on=["Id", "date"], how="left")

    if "Calories_y" in merged.columns and "Calories_x" in merged.columns:
        merged = merged.drop(columns=["Calories_y"]).rename(columns={"Calories_x": "Calories"})

    for col in ["LoggedActivitiesDistance", "SedentaryActiveDistance"]:
        if col in merged.columns:
            merged = merged.drop(columns=col)

    merged = merged[merged["Calories"] > 0].copy()
    merged["WeightKg"] = merged.groupby("Id")["WeightKg"].ffill().bfill()
    merged["BMI"] = merged.groupby("Id")["BMI"].ffill().bfill()
    merged["MinutesAsleep"] = merged["MinutesAsleep"].fillna(0)
    merged["AvgHeartRate"] = merged.groupby("Id")["AvgHeartRate"].transform(lambda s: s.fillna(s.mean()))
    merged["HoursAsleep"] = merged["MinutesAsleep"] / 60
    merged["TotalActiveMinutes"] = (
        merged["VeryActiveMinutes"] + merged["FairlyActiveMinutes"] + merged["LightlyActiveMinutes"]
    )
    merged["ActiveFraction"] = merged["TotalActiveMinutes"] / 1440
    merged["SleepCategory"] = pd.cut(
        merged["HoursAsleep"],
        bins=[-0.1, 6, 9, 24],
        labels=["Under-slept", "Normal", "Over-slept"],
    )

    def classify_activity(row):
        if row["VeryActiveMinutes"] >= 30:
            return "Very Active"
        if row["FairlyActiveMinutes"] >= 30:
            return "Fairly Active"
        if row["LightlyActiveMinutes"] >= 30:
            return "Lightly Active"
        return "Sedentary"

    merged["ActivityLevel"] = merged.apply(classify_activity, axis=1)
    merged["date"] = pd.to_datetime(merged["date"])

    return merged, hourly_steps, hourly_calories, hourly_intensity, heart_rate


@st.cache_data
def get_user_class(activity_df: pd.DataFrame) -> pd.DataFrame:
    counts = activity_df["Id"].value_counts().rename_axis("Id").reset_index(name="freq")
    counts["Class"] = np.where(
        counts["freq"] <= 10,
        "Light user",
        np.where(counts["freq"] <= 15, "Moderate user", "Heavy user"),
    )
    return counts[["Id", "Class", "freq"]]


@st.cache_data
def get_block_averages(hourly_steps: pd.DataFrame, hourly_calories: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    order = ["0-4", "4-8", "8-12", "12-16", "16-20", "20-24"]

    def blocks(hour: int) -> str:
        if 0 <= hour < 4:
            return "0-4"
        if 4 <= hour < 8:
            return "4-8"
        if 8 <= hour < 12:
            return "8-12"
        if 12 <= hour < 16:
            return "12-16"
        if 16 <= hour < 20:
            return "16-20"
        return "20-24"

    steps = hourly_steps.copy()
    steps["ActivityHour"] = pd.to_datetime(steps["ActivityHour"])
    steps["block"] = steps["ActivityHour"].dt.hour.apply(blocks)
    avg_steps = steps.groupby("block")["StepTotal"].mean().reindex(order)

    cals = hourly_calories.copy()
    cals["ActivityHour"] = pd.to_datetime(cals["ActivityHour"])
    cals["block"] = cals["ActivityHour"].dt.hour.apply(blocks)
    avg_cals = cals.groupby("block")["Calories"].mean().reindex(order)

    sleep = merged.copy()
    sleep["sleep_start_block"] = pd.to_datetime(sleep["ActivityDate"]).dt.hour.apply(blocks)
    avg_sleep = sleep.groupby("sleep_start_block")["MinutesAsleep"].mean().reindex(order)

    return pd.DataFrame(
        {
            "Steps": avg_steps,
            "Calories": avg_cals,
            "SleepMinutes": avg_sleep,
        }
    )


@st.cache_data
def get_heart_and_intensity(user_id: int, db_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(db_path)
    heart = pd.read_sql_query(
        f"SELECT Id, Time, Value FROM heart_rate WHERE Id = {int(user_id)}", conn
    )
    intensity = pd.read_sql_query(
        f"SELECT Id, ActivityHour, TotalIntensity FROM hourly_intensity WHERE Id = {int(user_id)}", conn
    )
    conn.close()

    if not heart.empty:
        heart["Time"] = pd.to_datetime(heart["Time"])
    if not intensity.empty:
        intensity["ActivityHour"] = pd.to_datetime(intensity["ActivityHour"])
    return heart, intensity


activity = load_activity_csv(CSV_PATH)
merged, hourly_steps, hourly_calories, hourly_intensity, heart_rate = load_merged_data(CSV_PATH, DB_PATH)
user_class_df = get_user_class(activity)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Participant details", "Sleep analysis"])

min_date = merged["date"].min().date()
max_date = merged["date"].max().date()
selected_dates = st.sidebar.date_input("Select date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

if isinstance(selected_dates, tuple) or isinstance(selected_dates, list):
    if len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = end_date = selected_dates[0]
else:
    start_date = end_date = selected_dates

filtered = merged[(merged["date"] >= pd.to_datetime(start_date)) & (merged["date"] <= pd.to_datetime(end_date))].copy()
user_ids = sorted(filtered["Id"].dropna().unique().tolist())
selected_user = st.sidebar.selectbox("Select participant ID", user_ids)

st.title("Fitbit Study Dashboard")
st.caption(f"Data range shown: {pd.to_datetime(start_date).date()} to {pd.to_datetime(end_date).date()}")

if page == "Overview":
    pop_users = filtered["Id"].nunique()
    avg_steps = filtered["TotalSteps"].mean()
    avg_calories = filtered["Calories"].mean()
    avg_sleep = filtered["HoursAsleep"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Participants", int(pop_users))
    c2.metric("Avg daily steps", f"{avg_steps:,.0f}")
    c3.metric("Avg daily calories", f"{avg_calories:,.0f}")
    c4.metric("Avg sleep hours", f"{avg_sleep:.2f}")

    st.subheader("Total distance by participant")
    distance_by_user = filtered.groupby("Id")["TotalDistance"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    distance_by_user.plot(kind="bar", ax=ax)
    ax.set_xlabel("Participant ID")
    ax.set_ylabel("Total distance")
    ax.set_title("Total distance by participant")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    left, right = st.columns(2)

    with left:
        st.subheader("Average steps over time")
        daily_steps = filtered.groupby("date")["TotalSteps"].mean()
        st.line_chart(daily_steps)

    with right:
        st.subheader("Activity level distribution")
        activity_dist = filtered["ActivityLevel"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(activity_dist.index, activity_dist.values)
        ax2.set_ylabel("Number of records")
        ax2.set_xlabel("Activity level")
        ax2.set_title("Distribution of activity levels")
        plt.xticks(rotation=20)
        plt.tight_layout()
        st.pyplot(fig2)

    st.subheader("Most active participants")
    st.bar_chart(distance_by_user.head(10))

    st.subheader("Population summary table")
    summary_table = filtered.groupby("Id")[["TotalDistance", "TotalSteps", "Calories", "HoursAsleep"]].mean().reset_index()
    st.dataframe(summary_table, use_container_width=True)

elif page == "Participant details":
    user_df = filtered[filtered["Id"] == selected_user].copy().sort_values("date")
    class_row = user_class_df[user_class_df["Id"] == selected_user]
    user_class = class_row["Class"].iloc[0] if not class_row.empty else "Unknown"

    st.header(f"Participant {selected_user}")
    st.write(f"Usage class: **{user_class}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total distance", f"{user_df['TotalDistance'].sum():.2f}")
    c2.metric("Avg steps", f"{user_df['TotalSteps'].mean():,.0f}")
    c3.metric("Avg calories", f"{user_df['Calories'].mean():,.0f}")
    c4.metric("Avg sleep hours", f"{user_df['HoursAsleep'].mean():.2f}")

    left, right = st.columns(2)

    with left:
        st.subheader("Distance over time")
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.plot(user_df["date"], user_df["TotalDistance"])
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Distance")
        ax3.set_title("Daily distance")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig3)

    with right:
        st.subheader("Calories over time")
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        ax4.plot(user_df["date"], user_df["Calories"])
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Calories")
        ax4.set_title("Daily calories")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig4)

    st.subheader("Calories vs steps")
    fig5, ax5 = plt.subplots(figsize=(7, 4))
    ax5.scatter(user_df["TotalSteps"], user_df["Calories"], alpha=0.7)
    if len(user_df) >= 2:
        m, b = np.polyfit(user_df["TotalSteps"], user_df["Calories"], 1)
        x_line = np.linspace(user_df["TotalSteps"].min(), user_df["TotalSteps"].max(), 100)
        ax5.plot(x_line, m * x_line + b, linestyle="--")
    ax5.set_xlabel("Total steps")
    ax5.set_ylabel("Calories")
    ax5.set_title("Calories vs steps")
    plt.tight_layout()
    st.pyplot(fig5)

    st.subheader("Heart rate and exercise intensity")
    heart_df, intensity_df = get_heart_and_intensity(selected_user, DB_PATH)
    if heart_df.empty or intensity_df.empty:
        st.info("No heart-rate or intensity data available for this participant.")
    else:
        fig6, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
        axes[0].plot(heart_df["Time"], heart_df["Value"])
        axes[0].set_title("Heart rate over time")
        axes[0].set_ylabel("Heart rate")

        axes[1].plot(intensity_df["ActivityHour"], intensity_df["TotalIntensity"])
        axes[1].set_title("Total intensity over time")
        axes[1].set_ylabel("Intensity")
        axes[1].set_xlabel("Time")
        plt.tight_layout()
        st.pyplot(fig6)

    st.subheader("Participant data table")
    st.dataframe(
        user_df[[
            "date", "TotalSteps", "TotalDistance", "Calories", "HoursAsleep",
            "SedentaryMinutes", "TotalActiveMinutes", "AvgHeartRate"
        ]],
        use_container_width=True,
    )

else:
    st.header("Sleep analysis")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg sleep hours", f"{filtered['HoursAsleep'].mean():.2f}")
    c2.metric("Avg active minutes", f"{filtered['TotalActiveMinutes'].mean():.1f}")
    c3.metric("Avg sedentary minutes", f"{filtered['SedentaryMinutes'].mean():.1f}")

    st.subheader("Sleep vs active minutes")
    fig7, ax7 = plt.subplots(figsize=(8, 5))
    plot_df = filtered.dropna(subset=["HoursAsleep", "TotalActiveMinutes"])
    ax7.scatter(plot_df["TotalActiveMinutes"], plot_df["HoursAsleep"], alpha=0.5, edgecolors="k", linewidths=0.3)
    if len(plot_df) >= 2:
        m, b = np.polyfit(plot_df["TotalActiveMinutes"], plot_df["HoursAsleep"], 1)
        x_line = np.linspace(plot_df["TotalActiveMinutes"].min(), plot_df["TotalActiveMinutes"].max(), 100)
        ax7.plot(x_line, m * x_line + b, color="red", linestyle="--")
        corr = plot_df["TotalActiveMinutes"].corr(plot_df["HoursAsleep"])
        ax7.annotate(f"r = {corr:.2f}", xy=(0.05, 0.93), xycoords="axes fraction")
    ax7.set_xlabel("Total active minutes")
    ax7.set_ylabel("Hours asleep")
    ax7.set_title("Relationship between activity and sleep")
    plt.tight_layout()
    st.pyplot(fig7)

    st.subheader("Sleep vs sedentary minutes")
    fig8, ax8 = plt.subplots(figsize=(8, 5))
    plot_df2 = filtered.dropna(subset=["HoursAsleep", "SedentaryMinutes"])
    ax8.scatter(plot_df2["SedentaryMinutes"], plot_df2["HoursAsleep"], alpha=0.5, edgecolors="k", linewidths=0.3)
    if len(plot_df2) >= 2:
        m, b = np.polyfit(plot_df2["SedentaryMinutes"], plot_df2["HoursAsleep"], 1)
        x_line = np.linspace(plot_df2["SedentaryMinutes"].min(), plot_df2["SedentaryMinutes"].max(), 100)
        ax8.plot(x_line, m * x_line + b, color="red", linestyle="--")
        corr = plot_df2["SedentaryMinutes"].corr(plot_df2["HoursAsleep"])
        ax8.annotate(f"r = {corr:.2f}", xy=(0.05, 0.93), xycoords="axes fraction")
    ax8.set_xlabel("Sedentary minutes")
    ax8.set_ylabel("Hours asleep")
    ax8.set_title("Relationship between sedentary time and sleep")
    plt.tight_layout()
    st.pyplot(fig8)

    left, right = st.columns(2)

    with left:
        st.subheader("Sleep category distribution")
        sleep_dist = filtered["SleepCategory"].value_counts(dropna=False)
        fig9, ax9 = plt.subplots(figsize=(6, 4))
        ax9.bar(sleep_dist.index.astype(str), sleep_dist.values)
        ax9.set_xlabel("Sleep category")
        ax9.set_ylabel("Count")
        ax9.set_title("Sleep quality categories")
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig9)

    with right:
        st.subheader("Time-of-day patterns")
        block_df = get_block_averages(hourly_steps, hourly_calories, filtered)
        metric_choice = st.selectbox("Metric by 4-hour block", ["Steps", "Calories", "SleepMinutes"])
        fig10, ax10 = plt.subplots(figsize=(7, 4))
        ax10.bar(block_df.index, block_df[metric_choice])
        ax10.set_xlabel("4-hour block")
        ax10.set_ylabel(metric_choice)
        ax10.set_title(f"Average {metric_choice} by 4-hour block")
        plt.tight_layout()
        st.pyplot(fig10)

    st.subheader("Selected participant sleep profile")
    sleep_user = filtered[filtered["Id"] == selected_user].sort_values("date")
    fig11, ax11 = plt.subplots(figsize=(9, 4))
    ax11.plot(sleep_user["date"], sleep_user["HoursAsleep"])
    ax11.set_xlabel("Date")
    ax11.set_ylabel("Hours asleep")
    ax11.set_title(f"Sleep over time for participant {selected_user}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig11)