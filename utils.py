
"""
Utility functions for the Quantified Self sleep, stress, and academic pressure project.
CPSC 222, Spring 2026
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import stats

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


HEALTH_FEATURES = [
    "Sleep Analysis [Total] (hr)",
    "Apple Exercise Time (min)",
    "Step Count (steps)",
    "Respiratory Rate (count/min)"
]

STRESS_FEATURES = [
    "Exam Day",
    "Assignment Due",
    "Mood Score"
]


def load_data(health_filename, stress_filename):
    """Loads the Apple Health and daily stress CSV files."""
    health_df = pd.read_csv(health_filename)
    stress_df = pd.read_csv(stress_filename)
    return health_df, stress_df


def clean_health_data(health_df):
    """Cleans the Apple Health data and keeps only the selected health features."""
    health_clean = health_df.copy()
    health_clean["Date/Time"] = pd.to_datetime(health_clean["Date/Time"], errors="coerce")
    health_clean["Date"] = health_clean["Date/Time"].dt.date

    keep_cols = ["Date"] + HEALTH_FEATURES
    keep_cols = [col for col in keep_cols if col in health_clean.columns]
    health_clean = health_clean[keep_cols]

    for col in HEALTH_FEATURES:
        if col in health_clean.columns:
            health_clean[col] = pd.to_numeric(health_clean[col], errors="coerce")

    return health_clean


def clean_stress_data(stress_df):
    """Cleans the stress log and creates numeric variables for modeling."""
    stress_clean = stress_df.copy()
    stress_clean["Date/Time"] = pd.to_datetime(stress_clean["Date/Time"], errors="coerce")
    stress_clean["Date"] = stress_clean["Date/Time"].dt.date

    for col in ["Stress Level", "Exam Day", "Assignment Due", "Mood"]:
        stress_clean[col] = stress_clean[col].astype(str).str.strip().str.lower()

    stress_clean["Stress Label"] = stress_clean["Stress Level"].map({"low": 0, "high": 1})
    stress_clean["Exam Day"] = stress_clean["Exam Day"].map({"no": 0, "yes": 1})
    stress_clean["Assignment Due"] = stress_clean["Assignment Due"].map({"no": 0, "yes": 1})

    mood_map = {
        "exhausted": 1,
        "tired": 2,
        "okay": 3,
        "good": 4
    }
    stress_clean["Mood Score"] = stress_clean["Mood"].map(mood_map)

    keep_cols = [
        "Date",
        "Stress Level",
        "Stress Label",
        "Exam Day",
        "Assignment Due",
        "Mood",
        "Mood Score"
    ]
    return stress_clean[keep_cols]


def merge_data(health_clean, stress_clean):
    """Merges the health and stress data tables by date."""
    return pd.merge(health_clean, stress_clean, on="Date", how="inner")


def add_week_column(df):
    """Adds a week number column for aggregation."""
    output_df = df.copy()
    output_df["Date"] = pd.to_datetime(output_df["Date"])
    output_df["Week"] = output_df["Date"].dt.isocalendar().week.astype(int)
    return output_df


def make_stress_pie_chart(df):
    """Creates a pie chart of high vs low stress days."""
    stress_counts = df["Stress Level"].value_counts()

    plt.figure(figsize=(6, 6))
    plt.pie(stress_counts, labels=stress_counts.index, autopct="%1.1f%%", startangle=90)
    plt.title("Distribution of High-Stress and Low-Stress Days")
    plt.show()


def make_avg_sleep_exam_bar(df):
    """Creates a bar chart comparing average sleep on exam and non-exam days."""
    sleep_by_exam = df.groupby("Exam Day")["Sleep Analysis [Total] (hr)"].mean()
    labels = ["No Exam", "Exam"]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, sleep_by_exam)
    plt.title("Average Sleep on Exam Days vs. Non-Exam Days")
    plt.xlabel("Exam Day")
    plt.ylabel("Average Sleep (hours)")
    plt.show()

def run_t_test_by_stress(df, column):
    """Runs a Welch two-sample t-test comparing high-stress and low-stress days."""
    high_values = df.loc[df["Stress Label"] == 1, column].dropna()
    low_values = df.loc[df["Stress Label"] == 0, column].dropna()

    t_stat, p_value = stats.ttest_ind(high_values, low_values, equal_var=False)

    print(f"High-stress mean {column}: {high_values.mean():.3f}")
    print(f"Low-stress mean {column}: {low_values.mean():.3f}")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.4f}")

    return t_stat, p_value


def run_t_test_by_exam(df, column):
    """Runs a Welch two-sample t-test comparing exam and non-exam days."""
    exam_values = df.loc[df["Exam Day"] == 1, column].dropna()
    non_exam_values = df.loc[df["Exam Day"] == 0, column].dropna()

    t_stat, p_value = stats.ttest_ind(exam_values, non_exam_values, equal_var=False)

    print(f"Exam-day mean {column}: {exam_values.mean():.3f}")
    print(f"Non-exam-day mean {column}: {non_exam_values.mean():.3f}")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.4f}")

    return t_stat, p_value
