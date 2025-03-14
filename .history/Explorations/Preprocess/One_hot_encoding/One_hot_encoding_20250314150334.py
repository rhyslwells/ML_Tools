import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load and preprocess data with pd.get_dummies (Dummy variables)
def load_and_preprocess_dummies():
    df = pd.read_csv("..\..\..\Datasets\homeprices.csv")
    # Create dummy variables for the 'town' column
    dummies = pd.get_dummies(df['town'], drop_first=True)  # Avoid dummy variable trap
    df_dummies = pd.concat([df, dummies], axis='columns')
    df_dummies = df_dummies.drop(['town'], axis='columns')

    # Split the dataset into features (X) and target (y)
    X = df_dummies.drop('price', axis='columns')
    y = df_dummies['price']
    
    return X, y

# Load and preprocess data with LabelEncoder
def load_and_preprocess_labelencoder():
    df = pd.read_csv("..\..\..\Datasets\homeprices.csv")
    # Apply LabelEncoder for 'town' column to convert it into numerical labels
    le = LabelEncoder()
    df['town'] = le.fit_transform(df['town'])
    X = df[['town', 'area']].values
    y = df['price'].values

    return X, y

# Load and preprocess data with OneHotEncoder
def load_and_preprocess_onehotencoder():
    df = pd.read_csv("..\..\..\Datasets\homeprices.csv")
    # Apply OneHotEncoding to 'town' column using ColumnTransformer
    ct = ColumnTransformer([('town', OneHotEncoder(), ['town'])], remainder='passthrough')
    df_transformed = ct.fit_transform(df)
    
    X = df_transformed[:, 1:]  # Remove one column to avoid the dummy variable trap
    y = df['price']
    
    return X, y

# Train a linear regression model
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Make predictions and store them for comparison
def make_predictions(model, examples, ct=None):
    predictions = []
    for example in examples:
        # Transform the input using the same ColumnTransformer (ct)
        transformed_example = ct.transform([example])  # Apply the same transformation used during training
        prediction = model.predict(transformed_example)
        predictions.append(prediction[0])  # Store the predicted value
    return predictions


# Function to demonstrate predictions with all preprocessing techniques and compare them
# Function to demonstrate predictions with all preprocessing techniques and compare them
def main():
    # Example houses: 3400 sq ft in West Windsor, 2800 sq ft in Robbinsville
    examples_dummies = [
        [3400, 0, 1],  # 3400 sq ft in West Windsor (after dummy encoding)
        [2800, 1, 0]   # 2800 sq ft in Robbinsville (after dummy encoding)
    ]
    
    examples_labelencoder = [
        [0, 3400],  # West Windsor (after label encoding)
        [1, 2800]   # Robbinsville (after label encoding)
    ]
    
    examples_onehotencoder = [
        [3400, 1, 0],  # West Windsor (after one-hot encoding)
        [2800, 0, 1]   # Robbinsville (after one-hot encoding)
    ]
    
    # Using pd.get_dummies
    X_dummies, y_dummies = load_and_preprocess_dummies()
    model_dummies = train_model(X_dummies, y_dummies)
    predictions_dummies = make_predictions(model_dummies, examples_dummies, ct=None)

    # Using LabelEncoder
    X_labelencoder, y_labelencoder = load_and_preprocess_labelencoder()
    model_labelencoder = train_model(X_labelencoder, y_labelencoder)
    predictions_labelencoder = make_predictions(model_labelencoder, examples_labelencoder, ct=None)

    # Using OneHotEncoder
    X_onehotencoder, y_onehotencoder = load_and_preprocess_onehotencoder()
    model_onehotencoder = train_model(X_onehotencoder, y_onehotencoder)
    
    # Create the ColumnTransformer used for OneHotEncoder to transform the examples during prediction
    ct = ColumnTransformer([('town', OneHotEncoder(), ['town'])], remainder='passthrough')
    
    predictions_onehotencoder = make_predictions(model_onehotencoder, examples_onehotencoder, ct=ct)

    # Creating a DataFrame to compare the predictions
    comparison_df = pd.DataFrame({
        'Method': ['pd.get_dummies', 'pd.get_dummies', 'LabelEncoder', 'LabelEncoder', 'OneHotEncoder', 'OneHotEncoder'],
        'House': ['3400 sq ft in West Windsor', '2800 sq ft in Robbinsville', '3400 sq ft in West Windsor', '2800 sq ft in Robbinsville', '3400 sq ft in West Windsor', '2800 sq ft in Robbinsville'],
        'Prediction': predictions_dummies + predictions_labelencoder + predictions_onehotencoder
    })

    print("\nComparison of Predictions:")
    print(comparison_df)

if __name__ == "__main__":
    main()
