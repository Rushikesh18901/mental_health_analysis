import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

def load_clustered_data(filepath):
    return pd.read_csv(filepath)

def train_model(df, target_column='treatment'):
    # Check if target column exists
    if target_column not in df.columns:
        print(f"Target column '{target_column}' not found. Available columns: {list(df.columns)}")
        return None

    # Select only numerical features for X
    numerical_cols = df.select_dtypes(include=[float, int]).columns
    X = df[numerical_cols].drop(target_column, axis=1, errors='ignore')
    y = df[target_column]

    if X.empty:
        print("No numerical features available for modeling.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(random_state=42)
    params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
    grid_search = GridSearchCV(rf, params, cv=5)
    grid_search.fit(X_train, y_train)

    predictions = grid_search.predict(X_test)
    print(classification_report(y_test, predictions))

    return grid_search.best_estimator_

if __name__ == "__main__":
    df = load_clustered_data('../Data/processed/clustered_data.csv')
    model = train_model(df)
    # Save model if needed