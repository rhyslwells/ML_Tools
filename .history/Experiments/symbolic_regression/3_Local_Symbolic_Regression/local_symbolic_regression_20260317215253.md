# Project 3: Local Symbolic Regression Models

## Objective
Use local symbolic models to interpret different regions of the feature space in a random forest.

## Approach

### Method 1: Region Partitioning from the Forest
1. Train random forest
2. Extract leaf regions from trees
3. Cluster similar leaves into larger regions
4. Fit symbolic regression inside each region

Example region: $R_1 = \{x : x_1 < 3, x_2 > 1.5\}$ with model $y = 1.8x_1 + 0.4x_2$

### Method 2: Clustering the Feature Space
1. Cluster observations: $X \rightarrow k$ clusters
2. Fit symbolic regression inside each cluster
3. Create piecewise model combining local formulas

### Method 3: Neighbourhood Models (LIME-style)
1. For each query point, sample nearby points
2. Evaluate random forest predictions
3. Fit local symbolic regression
4. Result: $f_{SR}^{(x_0)}(x)$ approximates local behavior

## Implementation Strategy

### Design Considerations
- Typical design: 5–20 regions
- Apply symbolic complexity constraints
- Enforce minimum sample size per region to avoid overfitting
- Final hybrid model: $f(x) = \sum_{i=1}^{k} \mathbb{1}_{x \in R_i} f_{SR}^{(i)}(x)$

### Region Definition and Fitting
- [ ] Define method for identifying regions
- [ ] Implement region extraction/clustering
- [ ] Fit SR within each region
- [ ] Combine into piecewise predictor

### Evaluation Metrics
Within region $R_i$, evaluate approximation error:
$$E_i = \frac{1}{n_i} \sum_{x \in R_i} (f_{RF}(x) - f_{SR}^{(i)}(x))^2$$

- Low error = locally explainable behavior
- Per-region accuracy tracking
- Global accuracy on validation set

## Implementation Progress

### Scripts/Notebooks
- [ ] setup_data.py - Generate synthetic data
- [ ] train_rf.py - Train random forest
- [ ] extract_regions.py - Extract or cluster regions
- [ ] fit_local_sr.py - Fit SR in each region
- [ ] evaluation.py - Evaluate per-region and global accuracy
- [ ] visualizations.py - Plot regions and local formulas

### Key Findings
(Document results here as you work through the project)

## Dependencies
- scikit-learn (random forest, clustering)
- gplearn or PySR (symbolic regression)
- numpy, pandas (data handling)
- matplotlib, seaborn (visualization)

## Next Steps
1. Choose method for region identification
2. Set up data and RF training
3. Extract/create regions
4. Fit local symbolic models
5. Evaluate against RF predictions
