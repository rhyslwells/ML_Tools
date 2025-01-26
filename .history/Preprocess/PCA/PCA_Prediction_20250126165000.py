import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv(r".\Datasets\heart.csv")

# Remove outliers (values exceeding 3 standard deviations)
df = df[(df.Cholesterol <= df.Cholesterol.mean() + 3 * df.Cholesterol.std()) &
        (df.MaxHR <= df.MaxHR.mean() + 3 * df.MaxHR.std()) &
        (df.FastingBS <= df.FastingBS.mean() + 3 * df.FastingBS.std()) &
        (df.Oldpeak <= df.Oldpeak.mean() + 3 * df.Oldpeak.std()) &
        (df.RestingBP <= df.RestingBP.mean() + 3 * df.RestingBP.std())]

# Encode categorical variables
df['ExerciseAngina'] = df['ExerciseAngina'].map({'N': 0, 'Y': 1})
df['ST_Slope'] = df['ST_Slope'].map({'Down': 1, 'Flat': 2, 'Up': 3})
df['RestingECG'] = df['RestingECG'].map({'Normal': 1, 'ST': 2, 'LVH': 3})

# One-hot encode remaining categorical features
df = pd.get_dummies(df, drop_first=True)

# Split data into features and target variable
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Standardize the features
X_scaled = StandardScaler().fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=30)

# Train RandomForest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
print(f"Accuracy (without PCA): {model.score(X_test, y_test)}")

# Apply PCA to reduce dimensionality (retain 95% of variance)
pca = PCA(0.95)
X_pca = pca.fit_transform(X_scaled)

# Split PCA data into training and test sets
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=30)

# Train and evaluate RandomForest model with PCA data
model.fit(X_train_pca, y_train)
print(f"Accuracy (with PCA): {model.score(X_test_pca, y_test)}")
