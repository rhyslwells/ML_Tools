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
from sklearn.model_selection import ParameterGrid

# Function to define and train a model based on given hyperparameters
def train_model(X_train, y_train, X_val, y_val, X_test, y_test, config):
    # Define the learning rate scheduler
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * 0.1

    lr_callback = LearningRateScheduler(scheduler)

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

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
        callbacks=[lr_callback, early_stopping],
        verbose=0
    )

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Track the performance of the model
    performance = {
        'config': config,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'history': history  # Store the history object
    }

    return performance


# Experiment Tracking Setup
experiment_results = []

# Define parameter grid for grid search
param_grid = {
    'learning_rate': [0.01, 0.005],
    'batch_size': [32, 64],
    'epochs': [30],
    'dropout': [0.3, 0.5],
    'units': [[64, 32], [128, 64]],
    'activation': ['relu', 'tanh'],
    'l2_reg': [0.005, 0.01]
}

# Generate all combinations of parameters using ParameterGrid
grid = ParameterGrid(param_grid)

# Mock data for demonstration (replace this with your actual dataset)
X, y = make_classification(n_samples=1000, n_features=400, n_classes=2, random_state=42)
y = y.reshape(-1, 1)  # Ensure y is in the correct shape

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Track experiments using grid search
for config in grid:
    performance = train_model(X_train, y_train, X_val, y_val, X_test, y_test, config)
    experiment_results.append(performance)

# Convert results into a DataFrame for better visualization
results_df = pd.DataFrame([{
    'Learning Rate': result['config']['learning_rate'],
    'Batch Size': result['config']['batch_size'],
    'Epochs': result['config']['epochs'],
    'Dropout': result['config']['dropout'],
    'Units': result['config']['units'],
    'Activation': result['config']['activation'],
    'L2 Regularization': result['config']['l2_reg'],
    'Test Accuracy': result['test_accuracy'],
    'Test Loss': result['test_loss']
} for result in experiment_results])

# Display the results in a table (printed)
print(results_df)

# Find the top 5 configurations based on test accuracy
top_5_results = sorted(experiment_results, key=lambda x: x['test_accuracy'], reverse=True)[:5]

# Plotting Loss for Each of the Top 5 Configurations
plt.figure(figsize=(10, 6))
for i, result in enumerate(top_5_results):
    plt.plot(result['history'].history['loss'], label=f"Config {i+1} (LR: {result['config']['learning_rate']}, Acc: {result['test_accuracy']:.4f})")

plt.title("Training Loss per Epoch for Top 5 Configurations")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting Accuracy for Each of the Top 5 Configurations
plt.figure(figsize=(10, 6))
for i, result in enumerate(top_5_results):
    plt.plot(result['history'].history['accuracy'], label=f"Config {i+1} (LR: {result['config']['learning_rate']}, Acc: {result['test_accuracy']:.4f})")

plt.title("Training Accuracy per Epoch for Top 5 Configurations")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting Test Accuracy for Each of the Top 5 Configurations
plt.figure(figsize=(10, 6))
plt.bar([f"Config {i+1}" for i in range(len(top_5_results))], 
        [result['test_accuracy'] for result in top_5_results], color='lightblue')
plt.title('Test Accuracy for Top 5 Configurations')
plt.xlabel('Experiment Config')
plt.ylabel('Test Accuracy')
plt.show()

# Plotting Test Loss for Each of the Top 5 Configurations
plt.figure(figsize=(10, 6))
plt.bar([f"Config {i+1}" for i in range(len(top_5_results))], 
        [result['test_loss'] for result in top_5_results], color='lightcoral')
plt.title('Test Loss for Top 5 Configurations')
plt.xlabel('Experiment Config')
plt.ylabel('Test Loss')
plt.show()

# Sort the experiment results based on test accuracy in descending order
sorted_results = sorted(experiment_results, key=lambda x: x['test_accuracy'], reverse=True)

# Define the number of top configurations you want (e.g., top 5)
top_n = 5

# Get the top N configurations based on test accuracy
top_results = sorted_results[:top_n]

# Print the top configurations based on test accuracy
print("\nTop 5 Best Configurations Based on Test Accuracy:")
for i, result in enumerate(top_results):
    print(f"\nTop {i+1}:")
    print(f"Learning Rate: {result['config']['learning_rate']}")
    print(f"Batch Size: {result['config']['batch_size']}")
    print(f"Epochs: {result['config']['epochs']}")
    print(f"Dropout: {result['config']['dropout']}")
    print(f"Units: {result['config']['units']}")
    print(f"Activation: {result['config']['activation']}")
    print(f"L2 Regularization: {result['config']['l2_reg']}")
    print(f"Test Accuracy: {result['test_accuracy']:.4f}")
    print(f"Test Loss: {result['test_loss']:.4f}")
