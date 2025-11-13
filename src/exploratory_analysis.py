import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def load_processed_data(filepath):
    return pd.read_csv(filepath)

def visualize_data(df):
    # Select only numerical columns for correlation
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numerical_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # Simple scatter plot instead of pairplot to avoid memory issues
    if 'Age' in df.columns and 'work_interfere_Often' in df.columns:
        plt.figure(figsize=(8, 6))
        plt.scatter(df['Age'], df['work_interfere_Often'])
        plt.xlabel('Age')
        plt.ylabel('Work Interference Often')
        plt.title('Age vs Work Interference')
        plt.show()

    # Interactive plot with Plotly
    if 'Age' in df.columns and 'work_interfere_Often' in df.columns:
        fig = px.scatter(df, x='Age', y='work_interfere_Often', color='Gender_Male')
        fig.show()

if __name__ == "__main__":
    df = load_processed_data('../Data/processed/processed_data.csv')
    visualize_data(df)