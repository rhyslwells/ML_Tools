# Symbolic Regression Project Ideas

## 1. Foundation: Decision Tree vs Symbolic Regression

**Objective:** Compare interpretability and accuracy between model types.

**Working Document:** [decision_tree_vs_sr.md](1_Decision_Tree_vs_SR/decision_tree_vs_sr.md)

- Generate synthetic data with a known symbolic relationship (e.g., $y = 2x^2 + 3x + 1$)
- Train both a decision tree and symbolic regression model on this data
- Compare predictions and examine how well each discovers the underlying formula
- Analyze which approach better recovers the true symbolic form

## 2. Random Forest vs Global Symbolic Regression

**Objective:** Test whether a single symbolic model can approximate ensemble behavior.

**Working Document:** [random_forest_vs_global_sr.md](2_Random_Forest_vs_Global_SR/random_forest_vs_global_sr.md)

- Train a random forest on synthetic data with a known relationship
- Fit a global symbolic regression model to match the random forest's predictions
- Compare the symbolic formula discovered to the actual underlying relationship
- Evaluate prediction accuracy across both models

## 3. Local Symbolic Regression Models

**Objective:** Use local symbolic models to interpret different regions of the feature space.

**Working Document:** [local_symbolic_regression.md](3_Local_Symbolic_Regression/local_symbolic_regression.md)

- Divide the random forest's prediction space into regions
- Fit local symbolic regression models for each region
- Compare local symbolic predictions to random forest predictions within those regions
- Identify which equations govern behavior in different areas of the input space

### Methods for Local Symbolic Regression

**Conceptual Structure**

Instead of fitting one global symbolic expression $f_{SR}(x)$, construct local symbolic models $f_{SR}^{(i)}(x)$ for regions $R_i$ of the feature space:

$$f(x) = \begin{cases} f_{SR}^{(1)}(x) & x \in R_1 \\ f_{SR}^{(2)}(x) & x \in R_2 \\ \dots \end{cases}$$

This approach works better because random forests inherently model local structure through piecewise regions.

**Method 1: Region Partitioning from the Forest**

1. Train random forest
2. Extract leaf regions from trees
3. Cluster similar leaves into larger regions
4. Fit symbolic regression inside each region

Example region: $R_1 = \{x : x_1 < 3, x_2 > 1.5\}$ with model $y = 1.8x_1 + 0.4x_2$

**Method 2: Clustering the Feature Space**

1. Cluster observations: $X \rightarrow \text{k clusters}$
2. Fit symbolic regression inside each cluster

Example:
- Cluster 1: $y = 2x_1 + 0.5x_2$
- Cluster 2: $y = \log(x_1) + 3x_3$

**Method 3: Neighbourhood Models (LIME-style)**

For a given point $x_0$:
1. Sample points near $x_0$
2. Evaluate random forest predictions
3. Fit symbolic regression locally

Result: $f_{SR}^{(x_0)}(x)$ approximates the local gradient structure.

### Advantages

- **Better approximation**: Local models capture simpler relationships than global models for highly nonlinear forests
- **Interpretability**: Obtain context-dependent rules (e.g., "If $x_1 < 3$ and $x_2 > 1$: $y \approx 2x_1 + 0.5x_2$")
- **Drift detection**: Monitor changes in regional formulas $f_{SR,t}^{(i)}(x)$ to reveal local concept drift

### Evaluation

Within region $R_i$, evaluate approximation error:
$$E_i = \frac{1}{n_i} \sum_{x \in R_i} (f_{RF}(x) - f_{SR}^{(i)}(x))^2$$

Low error means the forest behavior is locally explainable.

### Design Considerations

- Typical design: 5–20 regions
- Apply symbolic complexity constraints
- Enforce minimum sample size per region to avoid overfitting
- Final hybrid model: $f(x) = \sum_{i=1}^{k} \mathbf{1}_{x \in R_i} f_{SR}^{(i)}(x)$

## 4. Temporal Drift Monitoring

**Objective:** Track how symbolic models change when data distributions shift over time to detect and interpret concept drift.

**Working Document:** [temporal_drift_monitoring.md](4_Temporal_Drift_Monitoring/temporal_drift_monitoring.md)

- Set up a random forest that retrains periodically on streaming data
- Fit global symbolic models at each time step
- Track how the discovered symbolic formulas change over time
- Use formula changes as an interpretable drift detection signal that shows which regions/equations are evolving

### Problem Formulation

At each time step $t$ (e.g., daily):
- New dataset: $D_t = \{(x_i, y_i)\}_{i=1}^{n_t}$
- Fit symbolic regression: $f_t(x) = \text{symbolic_model}(D_t)$
- Objective: Determine whether the functional relationship $y = f(x)$ has changed over time

Concept drift occurs when: $f_t(x) \neq f_{t-k}(x)$ in either structure or parameters.

### Monitoring Pipeline

**Step 1: Fit Daily Symbolic Model**

For each day: $f_t = SR(D_t)$

Store:
- Expression tree
- Coefficients
- Training error

**Step 2: Evaluate Historical Performance**

Evaluate new formula on previous datasets: $E_{t,k} = \text{error}(f_t, D_{t-k})$

| model | data | error |
|-------|------|-------|
| $f_t$ | $D_t$ | 0.05 |
| $f_t$ | $D_{t-1}$ | 0.30 |
| $f_{t-1}$ | $D_t$ | 0.28 |

Large cross-error indicates drift.

**Step 3: Expression Similarity**

Compute structural similarity between expressions using:
- Tree edit distance
- Symbol overlap
- Functional equivalence test

Similarity score: $S(f_t, f_{t-1}) \in [0,1]$. Low similarity ⇒ structural drift.

**Step 4: Coefficient Monitoring**

If structure is identical, track coefficient vectors $\beta_t$ and monitor:
$$||\beta_t - \beta_{t-1}||$$
or use CUSUM / EWMA monitoring.

### Alternative Approaches

**Weighted Formula Ensemble**

Construct weighted predictor from daily models:
$$f^*(x) = \sum_{k=0}^{K} w_k f_{t-k}(x)$$

Example: Exponential decay weights $w_k = \lambda^k$ (normalized)

Benefit: Recent formulas influence prediction more while older formulas stabilize noise.

**Sliding Window Approach (Often Better)**

Instead of fitting separate daily models, fit on sliding window data:
$$D_{window} = \bigcup_{k=0}^{K} D_{t-k}$$

Then: $f_t = SR(D_{window})$

Advantages: Reduces noise, drift detection becomes smoother.

**Robust Formula Archive Method**

Maintain a formula archive $\mathcal{F} = \{f_1, f_2, ..., f_t\}$ and evaluate each on latest data:

| formula | error on $D_t$ |
|---------|----------------|
| $f_{t-5}$ | 0.04 |
| $f_{t-3}$ | 0.05 |
| $f_{t-1}$ | 0.22 |
| $f_t$ | 0.04 |

Interpretation: Old formula still works → no drift; only new formula works → drift occurred.

### Direct Functional Comparison

Instead of comparing formulas, compare function outputs:
$$\Delta = \frac{1}{m}\sum |f_t(x_i) - f_{t-1}(x_i)|$$

Large $\Delta$ ⇒ drift. This avoids symbolic comparison complexity.

### Practical Architecture

```
Daily pipeline
--------------
1. Ingest data D_t
2. Fit symbolic regression -> f_t
3. Store:
   - formula
   - coefficients
   - training error
4. Evaluate:
   - f_t on historical data
   - historical formulas on D_t
5. Drift tests:
   - error change
   - formula similarity
   - coefficient shift
6. Update ensemble predictor
```

### Key Challenges

**High Variance of Symbolic Regression**

Small data changes often produce different expressions even if the underlying function is identical.

Mitigations:
- Complexity penalties
- Pareto front selection
- Model simplification
- Stability selection

### Practical Design

A robust approach:
1. Fit symbolic regression on rolling window
2. Maintain Pareto frontier of formulas
3. Track: error on holdout stream, expression similarity, coefficient change
4. Trigger drift if: $E_t(f_{t-k}) - E_t(f_t) > \delta$ for several $k$


