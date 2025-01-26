# %% [markdown]
# <h3>PCA Machine Learning Tutorial Exercise Solution</h3>

# %%
import pandas as pd

# https://www.kaggle.com/fedesoriano/heart-failure-prediction
df = pd.read_csv(Datasets/heart.csv")
df.head()

# %%
df.shape

# %%
df.describe()

# %% [markdown]
# <h3>Treat Outliers</h3>

# %%
df[df.Cholesterol>(df.Cholesterol.mean()+3*df.Cholesterol.std())]

# %%
df.shape

# %%
df1 = df[df.Cholesterol<=(df.Cholesterol.mean()+3*df.Cholesterol.std())]
df1.shape

# %%
df[df.MaxHR>(df.MaxHR.mean()+3*df.MaxHR.std())]

# %%
df[df.FastingBS>(df.FastingBS.mean()+3*df.FastingBS.std())]

# %%
df[df.Oldpeak>(df.Oldpeak.mean()+3*df.Oldpeak.std())]

# %%
df2 = df1[df1.Oldpeak<=(df1.Oldpeak.mean()+3*df1.Oldpeak.std())]
df2.shape

# %%
df[df.RestingBP>(df.RestingBP.mean()+3*df.RestingBP.std())]

# %%
df3 = df2[df2.RestingBP<=(df2.RestingBP.mean()+3*df2.RestingBP.std())]
df3.shape

# %%
df.ChestPainType.unique()

# %%
df.RestingECG.unique()

# %%
df.ExerciseAngina.unique()

# %%
df.ST_Slope.unique()

# %%
df4 = df3.copy()
df4.ExerciseAngina.replace(
    {
        'N': 0,
        'Y': 1
    },
    inplace=True)

df4.ST_Slope.replace(
    {
        'Down': 1,
        'Flat': 2,
        'Up': 3
    },
    inplace=True
)

df4.RestingECG.replace(
    {
        'Normal': 1,
        'ST': 2,
        'LVH': 3
    },
    inplace=True)

df4.head()

# %%
df5 = pd.get_dummies(df4, drop_first=True)
df5.head()

# %%
X = df5.drop("HeartDisease",axis='columns')
y = df5.HeartDisease

X.head()

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=30)

# %%
X_train.shape

# %%
X_test.shape

# %%
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)
model_rf.score(X_test, y_test)

# %% [markdown]
# <h3>Use PCA to reduce dimensions</h3>

# %%
X

# %%
from sklearn.decomposition import PCA

pca = PCA(0.95)
X_pca = pca.fit_transform(X)
X_pca

# %%
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=30)

# %%
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier()
model_rf.fit(X_train_pca, y_train)
model_rf.score(X_test_pca, y_test)


