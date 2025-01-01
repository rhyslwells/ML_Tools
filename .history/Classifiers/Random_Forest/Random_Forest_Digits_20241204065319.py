import pandas as pd
from sklearn.datasets import load_digits



# Data exploration

digits = load_digits()
dir(digits)

digits.keys()

print(digits.images[0])

%matplotlib inline
import matplotlib.pyplot as plt
plt.gray() 
for i in range(4):
    plt.matshow(digits.images[i]) 
df = pd.DataFrame(digits.data)

df.head()

df['target']
digits.target

df[0:12]

# Train and the model

X = df.drop('target',axis='columns')
y = df.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train, y_train)
model.score(X_test, y_test)
y_predicted = model.predict(X_test)

# Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))

sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')



