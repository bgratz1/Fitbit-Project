# Fitbit Data Analysis Dashboard

## Overview
This repo is a full analysis done on Fitbit data that was collected from 33 individuals from a 2016 Amazon survey. The goal of this project is to explore trends in activity, sleep, and health metrics, and display our findings on a dashboard through Streamlit.

The project is structured based on the assignments where each part goes towards the dashboard. The final result allows users to be able to interact and explore the data through trends in the data by user, time, and different actrive metrics.

---

## Repository Structure

- `dashboard.py`  
  Main Streamlit application. This file compiles all previous parts and analysis into a dashboard with multiple pages which include general statistics, individual user analysis, sleep analysis, and temporal patterns.

- `part_1.py`  
  Contains the initial exploratory data analysis using the `daily_activity.csv` dataset. Includes:
  - Basic inspection of the dataset
  - Visualization of total distance per user
  - Temporal plots of calories burned
  - Analysis of workout frequency by day of the week
  - Linear regression between steps and calories

- `part3.py`  
  SQLite database. Includes:
  - SQL queries to extract relevant tables
  - Computation of sleep duration
  - Regression analyses between sleep, activity, and sedentary behavior
  - Time aggregation into 4 hour blocks
  - Heart rate and intensity analysis

- `part_4.py`  
  Focuses on data wrangling and feature engineering. Includes:
  - Handling missing values
  - Merging datasets across tables
  - Aggregating data by user and time
  - Generating summary statistics used in the dashboard

- `utils/`  
  Contains supporting data and helper functions used throughout the project, including database access and preprocessing utilities.

- `GenAI.docx`  
  Documentation of generative AI usage throughout the project, as required by the assignment guidelines.

---


## How to Run the Dashboard

1. Install required packages:
   ```bash
   pip install streamlit pandas matplotlib numpy sqlite3 statsmodels
   streamlit run dashboard.py
1. Run dashboard:
   ```bash
   streamlit run dashboard.py
