import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import confusion_matrix,classification_report

iris = load_iris()
dir(iris)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()

df['target'] = iris.target
# df[0:12]

# **Train and the model and prediction**
X = df.drop('target',axis='columns')
y = df.target

#modeling
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train, y_train)


model.score(X_test, y_test)
y_predicted = model.predict(X_test)

# **Confusion Matrix**

cm = confusion_matrix(y_test, y_predicted)
print(classification_report(y_test, y_predicted))