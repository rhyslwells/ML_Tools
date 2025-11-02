# Get an example of random forest regressor

# Apply the random forest regressor a dataset. Then evaluate the model performance with the R-squared metric [[R squared]]
	import numpy as np
    import pandas as pd
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score

    np.random.seed(1234)
    # load dataset
    ds = fetch_california_housing()
    df = pd.DataFrame(ds['data'], columns = ds['feature_names'])
    df['target'] = ds['target']
    # dependent variables
    X = df.drop('target', axis=1)
    # independent variable
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor( n_estimators=100 )
    model.fit(X_train, y_train)
    model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    r2_score(y_test, y_pred)
    # you should get 0.8049

