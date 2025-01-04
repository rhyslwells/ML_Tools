# Import necessary libraries
from pycaret.datasets import get_data
from pycaret.classification import setup, compare_models, create_model, evaluate_model, save_model, load_model

# Step 1: Load Dataset
data = get_data('iris')  # Load Iris dataset
print(data.head())  # Display the first few rows of the dataset

# Step 2: Initialize the PyCaret Environment
clf = setup(data, target='species', session_id=123, silent=True, html=False)  
# session_id ensures reproducibility, silent=True avoids user prompts

# Step 3: Compare Models
best_model = compare_models()  # Automatically compares models and selects the best one
print(f"Best Model: {best_model}")

# Step 4: Create a Specific Model
dt_model = create_model('dt')  # Create a Decision Tree model
print(f"Decision Tree Model: {dt_model}")

# Step 5: Evaluate the Model
evaluate_model(dt_model)  # Interactive model evaluation (plots available in the browser)

# Step 6: Save the Model
save_model(dt_model, 'best_dt_model')  # Save the model to a file
print("Model saved successfully!")

# Step 7: Load the Saved Model (Optional)
loaded_model = load_model('best_dt_model')  # Load the saved model
print(f"Loaded Model: {loaded_model}")

# Step 8: Predict on New Data (Optional)
# Assuming new_data is a Pandas DataFrame containing the same features as the training set
new_data = data.sample(5)  # Example: Using a random sample from the dataset
new_data_without_target = new_data.drop(columns=['species'])  # Drop target column
predictions = loaded_model.predict(new_data_without_target)
print(f"Predictions: {predictions}")
