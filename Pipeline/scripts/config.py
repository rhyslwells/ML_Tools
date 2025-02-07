class Config:
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    MISSING_VALUE_STRATEGY = "mean"
    MODELS = {
        "logistic_regression": {"solver": "liblinear"},
        "random_forest": {"n_estimators": 100, "max_depth": 5}
    }
