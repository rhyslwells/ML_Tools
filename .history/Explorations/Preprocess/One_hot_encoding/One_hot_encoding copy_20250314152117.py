import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Function to load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Dataset Loaded:\n", df.head())
    return df

# Function to perform dummy encoding
def dummy_encoding(df):
    dummies = pd.get_dummies(df.town)
    print("\nDummy Variables Created for 'town' column:")
    print(dummies.columns)
    merged = pd.concat([df, dummies], axis='columns')
    merged = merged.drop(['town'], axis='columns')
    merged = merged.drop(['west windsor'], axis='columns')  # Dropping one dummy column to avoid trap
    print("\nMerged DataFrame after dummy encoding:")
    print(merged.head())
    return merged

# Function to perform Label Encoding
def label_encoding(df):
    le = LabelEncoder()
    df_le = df.copy()
    df_le.town = le.fit_transform(df_le.town)
    print("\nLabelEncoder Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
    print("\nTransformed DataFrame with Encoded Town Names:")
    print(df_le.head())
    return df_le

# Function to perform One-Hot Encoding
def one_hot_encoding(df):
    ct = ColumnTransformer([('town', OneHotEncoder(), [0])], remainder='passthrough')
    X = ct.fit_transform(df[['town', 'area']].values)
    X = X[:, 1:]  # Drop first column to avoid dummy variable trap
    print("\nTransformed Features after OneHotEncoding:\n", X)
    return X

# Function to train a Linear Regression model
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    print("\nLinear Regression Model Trained.")
    return model

# Function to make predictions
def make_predictions(model, X):
    return model.predict(X)

# Function to predict for specific examples
def predict_example(model, example):
    return model.predict([example])

# Main function to tie everything together
def main():
    file_path = "..\\..\\..\\Datasets\\homeprices.csv"
    df = load_data(file_path)
    
    # Prepare target variable
    y = df.price

    # 1. Dummy Encoding
    print("\nTesting Dummy Encoding...")
    df_dummy = dummy_encoding(df)
    X_dummy = df_dummy.drop('price', axis='columns')
    model_dummy = train_model(X_dummy, y)
    predictions_dummy = make_predictions(model_dummy, X_dummy)
    print("\nPredicted Prices (Dummy Encoding):", predictions_dummy)

    # 2. Label Encoding
    print("\nTesting Label Encoding...")
    df_label = label_encoding(df)
    X_label = df_label[['town', 'area']].values
    model_label = train_model(X_label, y)
    predictions_label = make_predictions(model_label, X_label)
    print("\nPredicted Prices (Label Encoding):", predictions_label)

    # 3. One-Hot Encoding
    print("\nTesting One-Hot Encoding...")
    X_onehot = one_hot_encoding(df)
    model_onehot = train_model(X_onehot, y)
    predictions_onehot = make_predictions(model_onehot, X_onehot)
    print("\nPredicted Prices (OneHot Encoding):", predictions_onehot)

    # Comparison of predictions
    print("\nComparison of predictions for a 3400 sq ft house:")
    example = [3400, 0, 0]  # Example: 3400 sq ft in West Windsor (Dummy Encoding)
    print("Dummy Encoding:", predict_example(model_dummy, example))
    print("Label Encoding:", predict_example(model_label, [2, 3400]))
    print("OneHot Encoding:", predict_example(model_onehot, [0, 1, 3400]))

    print("\nComparison of predictions for a 2800 sq ft house:")
    example2 = [2800, 0, 1]  # Example: 2800 sq ft in Robbinsville (Dummy Encoding)
    print("Dummy Encoding:", predict_example(model_dummy, example2))
    print("Label Encoding:", predict_example(model_label, [1, 2800]))
    print("OneHot Encoding:", predict_example(model_onehot, [1, 0, 2800]))

if __name__ == "__main__":
    main()
