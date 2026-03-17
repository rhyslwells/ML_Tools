# Project 2: Random Forest vs Global Symbolic Regression

## Objective
Test whether a single symbolic model can approximate ensemble behavior.

## Approach

### 1. Model Training
- Train a random forest on synthetic data with a known relationship
- Fit a global symbolic regression model to match the random forest's predictions
- Build SR model to predict RF outputs (not the ground truth directly)

### 2. Comparison
- Compare the symbolic formula discovered to the actual underlying relationship
- Evaluate prediction accuracy across both models
- Measure how well SR approximates RF behavior

### 3. Evaluation Metrics
- **RF vs Ground Truth**: How well does RF learn the true relationship?
- **SR vs RF**: How well does SR approximate the RF?
- **SR vs Ground Truth**: Does SR recover the underlying formula?
- **Formula Comparison**: Analyze structural differences between discovered and true formulas

## Implementation Progress

### Scripts/Notebooks
- [ ] setup_experiment.py - Data generation and RF training
- [ ] fit_symbolic_regression.py - Fit SR to RF predictions
- [ ] comparison_analysis.py - Compare formulas and accuracy
- [ ] visualizations.py - Plot RF vs SR vs ground truth

### Key Questions to Address
- How much does the symbolic model lose in approximating an ensemble?
- Can the ensemble's complexity be captured by a simple formula?
- What types of relationships are well-approximated vs poorly-approximated?

### Key Findings
(Document results here as you work through the project)

## Dependencies
- scikit-learn (random forest)
- gplearn or PySR (symbolic regression)
- numpy, pandas (data handling)
- matplotlib (visualization)

## Next Steps
1. Generate synthetic data with known relationship
2. Train random forest
3. Use grid search to optimize SR fitting to RF predictions
4. Analyze discovered formula vs truth
