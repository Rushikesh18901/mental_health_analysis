import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    # Handle missing values for Age (numerical)
    imputer = KNNImputer(n_neighbors=3)
    df[['Age']] = imputer.fit_transform(df[['Age']])

    # One-hot encode categorical variables
    cat_cols = ['Gender', 'no_employees', 'work_interfere']
    df_encoded = pd.get_dummies(df, columns=cat_cols)

    # Standard scaling of numerical features
    scaler = StandardScaler()
    df_encoded[['Age']] = scaler.fit_transform(df_encoded[['Age']])

    return df_encoded

if __name__ == "__main__":
    df = load_data('../Data/raw/survey.csv')
    df_processed = preprocess_data(df)
    df_processed.to_csv('../Data/processed/processed_data.csv', index=False)