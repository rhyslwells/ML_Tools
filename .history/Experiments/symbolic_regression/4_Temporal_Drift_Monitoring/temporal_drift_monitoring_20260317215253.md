# Project 4: Temporal Drift Monitoring

## Objective
Track how symbolic models change when data distributions shift over time to detect and interpret concept drift.

## Approach

### Problem Setup
At each time step $t$ (e.g., daily):
- New dataset: $D_t = \{(x_i, y_i)\}_{i=1}^{n_t}$
- Fit symbolic regression: $f_t(x) = \text{symbolic_model}(D_t)$
- Objective: Determine whether the functional relationship $y = f(x)$ has changed over time

Concept drift occurs when: $f_t(x) \neq f_{t-k}(x)$ in either structure or parameters.

### Monitoring Pipeline

**Step 1: Fit Daily Symbolic Model**
- For each time window: $f_t = SR(D_t)$
- Store: expression tree, coefficients, training error

**Step 2: Evaluate Historical Performance**
- Evaluate new formula on previous datasets: $E_{t,k} = \text{error}(f_t, D_{t-k})$
- Large cross-error indicates drift

**Step 3: Expression Similarity**
- Compute structural similarity between expressions
- Use: tree edit distance, symbol overlap, functional equivalence
- Similarity score: $S(f_t, f_{t-1}) \in [0,1]$. Low similarity ⇒ structural drift

**Step 4: Coefficient Monitoring**
- If structure is identical, track coefficient vectors $\beta_t$
- Monitor: $||\beta_t - \beta_{t-1}||$
- Use CUSUM or EWMA monitoring methods

### Alternative Strategies

**Sliding Window Approach (Recommended)**
- Fit on sliding window of recent data instead of single timepoint
- Advantages: Reduces noise, smoother drift detection
- Window size: 10-50 observations (depends on data frequency)

**Weighted Formula Ensemble**
- Construct predictor from recent models with exponential decay
- $f^*(x) = \sum_{k=0}^{K} w_k f_{t-k}(x)$ where $w_k = \lambda^k$

**Robust Formula Archive Method**
- Maintain archive of formulas: $\mathcal{F} = \{f_1, f_2, ..., f_t\}$
- Evaluate each formula on latest data
- Detect which formulas remain valid vs become stale

## Implementation Strategy

### Drift Detection Methods
- Error degradation: Track error of old formulas on new data
- Formula distance: Compute similarity between consecutive models
- Coefficient stability: Monitor parameter changes
- Functional comparison: $\Delta = \frac{1}{m}\sum |f_t(x_i) - f_{t-1}(x_i)|$

### Practical Architecture
```
Daily pipeline
--------------
1. Ingest data D_t
2. Fit symbolic regression -> f_t
3. Store: formula, coefficients, training error
4. Evaluate:
   - f_t on historical data
   - historical formulas on D_t
5. Drift tests:
   - error change
   - formula similarity
   - coefficient shift
6. Update ensemble predictor
```

## Implementation Progress

### Scripts/Notebooks
- [ ] generate_drift_data.py - Simulate temporal data with concept drift
- [ ] fit_temporal_models.py - Fit SR at each timestep
- [ ] drift_detection.py - Implement drift detection methods
- [ ] formula_comparison.py - Compare formulas across time
- [ ] evaluation.py - Evaluate drift detection performance
- [ ] visualizations.py - Plot drift timeline and formula changes

### Key Questions to Address
- How sensitive is each drift detection method?
- Can drift be detected before prediction accuracy degrades significantly?
- Which formulas characterize each phase of drift?
- How do local vs global drift patterns differ?

### Key Findings
(Document results here as you work through the project)

## Dependencies
- scikit-learn (random forest for baseline)
- gplearn or PySR (symbolic regression)
- numpy, pandas (data handling)
- matplotlib, seaborn (visualization)
- scipy (signal processing for CUSUM/EWMA)

## Next Steps
1. Generate synthetic data with temporal concept drift
2. Implement basic SR fitting loop
3. Build drift detection metrics
4. Compare detection methods
5. Visualize drift evolution over time
