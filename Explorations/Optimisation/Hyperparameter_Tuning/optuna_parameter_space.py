import optuna
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.regularizers import l2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.decomposition import PCA
from optuna.integration import TFKerasPruningCallback

# Function to define and train a model based on given hyperparameters
def train_model(X_train, y_train, X_val, y_val, X_test, y_test, config, trial):
    # Define the learning rate scheduler
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * 0.1

    lr_callback = LearningRateScheduler(scheduler)

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Pruning callback
    pruning_callback = TFKerasPruningCallback(trial, 'val_loss')

    # Build the model based on configuration
    model = Sequential([
        Dense(config['units'][0], activation=config['activation'], kernel_initializer='he_normal', kernel_regularizer=l2(config['l2_reg']), input_shape=(X_train.shape[1],)),
        Dropout(config['dropout']),
        Dense(config['units'][1], activation=config['activation'], kernel_initializer='he_normal', kernel_regularizer=l2(config['l2_reg'])),
        Dropout(config['dropout']),
        Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=config['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train the model with callbacks
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=[lr_callback, early_stopping, pruning_callback],
        verbose=0
    )

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    return test_loss, test_accuracy

# Optuna objective function
def objective(trial):
    # Hyperparameters to be tuned
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    epochs = 30  # Fixed number of epochs
    dropout = trial.suggest_uniform('dropout', 0.2, 0.5)
    units_1 = trial.suggest_int('units_1', 64, 128)
    units_2 = trial.suggest_int('units_2', 32, 128)
    activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
    l2_reg = trial.suggest_loguniform('l2_reg', 1e-6, 1e-2)

    # Mock data for demonstration (replace this with your actual dataset)
    X, y = make_classification(n_samples=1000, n_features=400, n_classes=2, random_state=42)
    y = y.reshape(-1, 1)  # Ensure y is in the correct shape

    # Split data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Configuration dictionary for model
    config = {
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'dropout': dropout,
        'units': [units_1, units_2],
        'activation': activation,
        'l2_reg': l2_reg
    }

    # Train and evaluate the model
    test_loss, test_accuracy = train_model(X_train, y_train, X_val, y_val, X_test, y_test, config, trial)

    # Return the objective value (we aim to maximize accuracy)
    return test_accuracy

# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='maximize')  # We are maximizing test accuracy
study.optimize(objective, n_trials=15)  # Run 15 trials for demonstration (increase trials for a more comprehensive search)

# Get the best trial
best_trial = study.best_trial
print("Best Trial:")
print(f"Test Accuracy: {best_trial.value}")
print(f"Best Hyperparameters: {best_trial.params}")

# Display the results
results_df = pd.DataFrame(study.trials_dataframe())
print(results_df)

# Plotting optimization process (accuracy over trials)
plt.figure(figsize=(10, 6))
plt.plot([trial.number for trial in study.trials], [trial.value for trial in study.trials], marker='o', linestyle='--', color='b')
plt.title("Test Accuracy per Trial")
plt.xlabel('Trial Number')
plt.ylabel('Test Accuracy')
plt.show()

# Visualize the parameter search space
plt.figure(figsize=(10, 6))
param_values = results_df[['params_learning_rate', 'params_batch_size', 'params_dropout', 'params_units_1', 'params_units_2', 'params_activation', 'params_l2_reg']]
param_values['accuracy'] = study.trials_dataframe()['value']
param_values.plot(x='params_learning_rate', y='accuracy', kind='scatter', color='blue', alpha=0.5)
plt.title("Hyperparameter Search with Test Accuracy")
plt.xlabel("Learning Rate")
plt.ylabel("Test Accuracy")
plt.show()

# Pairwise Plot (Scatter Matrix)
sns.pairplot(results_df[['params_learning_rate', 'params_batch_size', 'params_dropout', 'params_units_1', 'params_units_2', 'params_activation', 'params_l2_reg', 'value']], hue='value', palette='coolwarm')
plt.show()

# Contour Plot for Learning Rate vs Dropout vs Accuracy
# Ensure the pivot table is created correctly and fill missing values
pivot_df = results_df.pivot_table('value', 'params_learning_rate', 'params_dropout', aggfunc='mean')

# Fill missing values (NaN) using backward fill method (bfill)
pivot_df = pivot_df.fillna(method='bfill')  # Using backward fill for missing values

# Plot the contour plot
plt.figure(figsize=(10, 6))
contour = plt.contourf(pivot_df.columns, pivot_df.index, pivot_df.values, cmap='coolwarm', levels=50)
plt.colorbar(contour, label='Accuracy')
plt.xlabel('Learning Rate')
plt.ylabel('Dropout')
plt.title('Contour Plot: Learning Rate vs Dropout vs Accuracy')
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Create a scatter plot with 'params_learning_rate' and 'params_batch_size' on the x-axis and 'params_dropout' on the y-axis
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=results_df['params_learning_rate'], 
    y=results_df['params_dropout'], 
    hue=results_df['params_batch_size'],  # Use batch size as the hue for color coding
    palette='coolwarm',  # Color palette
    s=100,  # Adjust the size of the points for better visibility
)

# Add labels and title
plt.xlabel('Learning Rate & Batch Size')
plt.ylabel('Dropout')
plt.title('Learning Rate, Batch Size vs Dropout')

# Show the plot
plt.show()
