# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "openai",
#   "argparse",
#   "numpy",
#   "scipy",
#   "scikit-learn"
# ]
# ///

import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import requests

# Fetch the AI Proxy Token from environment variables
API_TOKEN = os.environ.get("AIPROXY_TOKEN")
if not API_TOKEN:
    raise EnvironmentError("AIPROXY_TOKEN environment variable is not set.")

PROXY_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

# Command-line argument setup
def parse_arguments():
    parser = argparse.ArgumentParser(description="Automated analysis and narration of a dataset.")
    parser.add_argument("dataset", type=str, help="Path to the CSV file to analyze.")
    return parser.parse_args()

# Load the dataset
def load_dataset(filepath):
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        raise FileNotFoundError(f"Error loading dataset: {e}")

# Basic dataset analysis
def basic_analysis(df):
    analysis = {
        "shape": df.shape,
        "columns": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "summary_statistics": df.describe(include='all').to_dict(),
    }
    return analysis

# Advanced analysis: Clustering
def cluster_analysis(df):
    numerical_df = df.select_dtypes(include=["number"]).dropna()
    if numerical_df.shape[1] > 1:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numerical_df)
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        df['Cluster'] = clusters
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=numerical_df.iloc[:, 0], y=numerical_df.iloc[:, 1], hue=clusters, palette="viridis")
        plt.title("K-Means Clustering")
        plt.savefig("cluster_analysis.png")
        plt.close()
        return "cluster_analysis.png"
    return None

# Advanced analysis: Regression
def regression_analysis(df):
    numerical_df = df.select_dtypes(include=["number"]).dropna()
    if numerical_df.shape[1] > 1:
        target = numerical_df.columns[-1]
        X = numerical_df.drop(columns=[target])
        y = numerical_df[target]
        model = LinearRegression()
        model.fit(X, y)
        coefficients = dict(zip(X.columns, model.coef_))
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(coefficients.keys()), y=list(coefficients.values()))
        plt.title("Regression Coefficients")
        plt.savefig("regression_analysis.png")
        plt.close()
        return "regression_analysis.png"
    return None

# Advanced analysis: Time Series
def time_series_analysis(df):
    if any(df.dtypes == "datetime64[ns]"):
        datetime_col = df.select_dtypes(include=["datetime64[ns]"]).columns[0]
        df.set_index(datetime_col, inplace=True)
        if df.index.is_all_dates:
            numerical_df = df.select_dtypes(include=["number"]).dropna()
            for col in numerical_df.columns:
                plt.figure(figsize=(12, 6))
                sns.lineplot(x=numerical_df.index, y=numerical_df[col])
                plt.title(f"Time Series Analysis: {col}")
                plt.savefig(f"time_series_{col}.png")
                plt.close()
            return [f"time_series_{col}.png" for col in numerical_df.columns]
    return []

# Generate visualizations
def create_visualizations(df):
    visuals = []
    # Correlation heatmap
    if df.select_dtypes(include=["number"]).shape[1] > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        plt.savefig("correlation_matrix.png")
        plt.close()
        visuals.append("correlation_matrix.png")

    # Advanced visualizations
    cluster_visual = cluster_analysis(df)
    if cluster_visual:
        visuals.append(cluster_visual)

    regression_visual = regression_analysis(df)
    if regression_visual:
        visuals.append(regression_visual)

    time_series_visuals = time_series_analysis(df)
    visuals.extend(time_series_visuals)

    return visuals

# Interact with the LLM
def interact_with_llm(prompt):
    try:
        headers = {
            "Authorization": f"Bearer {API_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are an expert data analyst."},
                {"role": "user", "content": prompt}
            ]
        }
        response = requests.post(PROXY_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Error interacting with LLM: {e}")

# Generate Markdown report
def generate_readme(analysis, visuals):
    with open("README.md", "w") as f:
        f.write("# Automated Data Analysis\n\n")
        f.write("## Summary of Analysis\n\n")
        f.write(f"**Dataset Shape**: {analysis['shape']}\n\n")
        f.write("### Missing Values\n\n")
        for col, count in analysis['missing_values'].items():
            f.write(f"- {col}: {count} missing values\n")
        f.write("\n### Summary Statistics\n\n")
        for col, stats in analysis['summary_statistics'].items():
            f.write(f"#### {col}\n")
            for stat, value in stats.items():
                f.write(f"- {stat}: {value}\n")
        f.write("\n### Visualizations\n\n")
        for visual in visuals:
            f.write(f"![{visual}]({visual})\n")

# Main script
def main():
    args = parse_arguments()
    df = load_dataset(args.dataset)
    analysis = basic_analysis(df)
    visuals = create_visualizations(df)

    prompt = "Provide a narrative analysis of the following data: \n" + str(analysis)
    narrative = interact_with_llm(prompt)

    generate_readme(analysis, visuals)

if __name__ == "__main__":
    main()
