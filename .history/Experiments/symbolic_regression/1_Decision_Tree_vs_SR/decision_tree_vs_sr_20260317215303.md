# Project 1: Decision Tree vs Symbolic Regression

## Objective
Compare interpretability and accuracy between decision tree and symbolic regression model types.

## Approach

### 1. Data Generation
- Generate synthetic data with a known symbolic relationship (e.g., $y = 2x^2 + 3x + 1$)
- Vary dataset size for robustness testing
- Consider multiple relationship types (polynomial, logarithmic, etc.)

### 2. Model Training
- Train a decision tree on synthetic data
- Train symbolic regression model on the same data
- Document hyperparameters and settings

### 3. Comparison Metrics
- **Accuracy**: MSE, R² score
- **Interpretability**: 
  - How well does each model recover the true formula?
  - Simplicity of discovered relationships
  - Ease of understanding predictions
- **Complexity**: Model size, number of parameters

### 4. Analysis
- Compare which approach better recovers the true symbolic form
- Evaluate prediction accuracy across both models
- Generate visualizations of predictions vs ground truth

## Implementation Progress

### Scripts/Notebooks
- [ ] data_generation.py - Synthetic data creation
- [ ] train_models.py - Decision tree and SR training
- [ ] evaluation.py - Metrics and comparisons
- [ ] visualizations.py - Plotting results

### Key Findings
(Document results here as you work through the project)

## Dependencies
- scikit-learn (decision trees)
- gplearn or PySR (symbolic regression)
- numpy, pandas (data handling)
- matplotlib (visualization)

## Next Steps
1. Choose specific symbolic relationships to test
2. Set up data generation pipeline
3. Train initial models
4. Analyze recovery of ground truth formulas
