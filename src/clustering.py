import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_processed_data(filepath):
    return pd.read_csv(filepath)

def perform_clustering(df, n_clusters=4):
    # Use available numerical features
    features = df[['Age']]  # Start with Age
    # Add work_interfere columns if they exist
    work_cols = [col for col in df.columns if col.startswith('work_interfere_')]
    if work_cols:
        features = pd.concat([features, df[work_cols]], axis=1)
    # Add other numerical columns if available
    numerical_cols = df.select_dtypes(include=[float, int]).columns
    for col in numerical_cols:
        if col not in features.columns and col != 'cluster':
            features[col] = df[col]

    pca = PCA(n_components=2)
    features_reduced = pca.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features_reduced)
    df['cluster'] = clusters

    score = silhouette_score(features_reduced, clusters)
    print("Silhouette Score:", score)
    return df

if __name__ == "__main__":
    df = load_processed_data('../Data/processed/processed_data.csv')
    df_clustered = perform_clustering(df)
    df_clustered.to_csv('../Data/processed/clustered_data.csv', index=False)