import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# create a dataframe with sample data
data = {'X1':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'X2':[11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'X3':[21, 22, 23, 24, 25, 26, 27, 28, 29, 30], 'y':[31, 32, 33, 34, 35, 36, 37, 38, 39, 40]}
df = pd.DataFrame(data) 
# split the data into training and test sets
X = df.drop('y', axis=1)
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# apply PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
# train a linear regression model
model = LinearRegression()
model.fit(X_train_pca, y_train)
# evaluate the model
y_pred = model.predict(X_test_pca)
r2_score(y_test, y_pred)
# you should get 0.9999