
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
    health_clean["Date/Time"] = pd.to_datetime(health_clean["Date/Time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
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
    stress_clean["Date/Time"] = pd.to_datetime(stress_clean["Date/Time"], format="%m/%d/%y %H:%M", errors="coerce")
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


def make_sleep_stress_boxplot(df):
    """Creates a boxplot comparing sleep by stress level."""
    plt.figure(figsize=(7, 5))
    df.boxplot(column="Sleep Analysis [Total] (hr)", by="Stress Level")
    plt.title("Sleep Distribution by Stress Level")
    plt.suptitle("")
    plt.xlabel("Stress Level")
    plt.ylabel("Sleep Analysis [Total] (hr)")
    plt.show()


def make_steps_sleep_scatter(df):
    """Creates a scatterplot comparing step count and sleep."""
    plt.figure(figsize=(7, 5))
    plt.scatter(df["Step Count (steps)"], df["Sleep Analysis [Total] (hr)"])
    plt.title("Step Count vs. Total Sleep")
    plt.xlabel("Step Count")
    plt.ylabel("Total Sleep (hours)")
    plt.show()


def make_sleep_line_graph(df):
    """Creates a line graph of sleep over time."""
    ordered_df = df.sort_values("Date")

    plt.figure(figsize=(12, 5))
    plt.plot(ordered_df["Date"], ordered_df["Sleep Analysis [Total] (hr)"], marker="o")
    plt.title("Total Sleep Over Time")
    plt.xlabel("Date")
    plt.ylabel("Total Sleep (hours)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def make_weekly_averages_line_graph(df):
    """Creates a line graph of weekly averages for selected health features."""
    week_df = add_week_column(df)

    weekly_avg = week_df.groupby("Week")[
        [
            "Sleep Analysis [Total] (hr)",
            "Apple Exercise Time (min)",
            "Step Count (steps)",
            "Respiratory Rate (count/min)"
        ]
    ].mean().reset_index()

    plt.figure(figsize=(10, 5))
    plt.plot(weekly_avg["Week"], weekly_avg["Sleep Analysis [Total] (hr)"], marker="o")
    plt.title("Average Total Sleep by Week")
    plt.xlabel("Week")
    plt.ylabel("Average Sleep (hours)")
    plt.xticks(weekly_avg["Week"])
    plt.show()

    return weekly_avg


def make_correlation_heatmap(df):
    """Creates a correlation heatmap for numeric project variables."""
    corr_cols = [
        "Sleep Analysis [Total] (hr)",
        "Apple Exercise Time (min)",
        "Step Count (steps)",
        "Respiratory Rate (count/min)",
        "Exam Day",
        "Assignment Due",
        "Mood Score",
        "Stress Label"
    ]

    corr_df = df[corr_cols].corr()

    plt.figure(figsize=(9, 7))
    plt.imshow(corr_df, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=90)
    plt.yticks(range(len(corr_df.columns)), corr_df.columns)
    plt.title("Correlation Heatmap of Health and Stress Variables")
    plt.tight_layout()
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


def prepare_classification_data(df):
    """Separates predictors and class label for classification."""
    feature_cols = HEALTH_FEATURES + STRESS_FEATURES
    X = df[feature_cols]
    y = df["Stress Label"]
    return X, y


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Fits a model and prints accuracy, classification report, and confusion matrix."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"{model_name} Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, predictions))
    print(f"{model_name} Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))


def make_knn_model():
    """Builds a kNN classifier pipeline with median imputation and scaling."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ])


def make_decision_tree_model():
    """Builds a decision tree classifier pipeline with median imputation."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", DecisionTreeClassifier(max_depth=4, random_state=42))
    ])


def make_random_forest_model():
    """Builds an optional random forest classifier pipeline with median imputation."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])


def plot_simple_decision_tree(df):
    """Plots a readable decision tree using the selected project features."""
    X, y = prepare_classification_data(df)

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X_imputed, y)

    plt.figure(figsize=(18, 8))
    plot_tree(
        tree,
        feature_names=X.columns,
        class_names=["low stress", "high stress"],
        filled=True,
        rounded=True
    )
    plt.title("Decision Tree for Predicting Stress Level")
    plt.show()
