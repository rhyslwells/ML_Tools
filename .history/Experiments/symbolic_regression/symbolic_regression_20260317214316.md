# Symbolic Regression

Symbolic regression attempts to discover an explicit functional form: $y = f(x)$

**Example output**: $y = 2.1x_1 + \log(x_2) - 0.5x_3^2$

The output is a closed-form mathematical expression typically represented as an expression tree composed of:
- Variables
- Constants
- Operators: $+$, $-$, $\times$, $\div$
- Nonlinear functions: $\log$, $\exp$, $\sin$, etc.

## Computational Cost

Very expensive due to:
- Search over expression space
- Genetic programming approach
- Large candidate population
- Training time scales poorly with feature count

## Extrapolation Behaviour

Because the model is a mathematical function, it can extrapolate beyond training data.

**Example**: If the model learns $y = 2x$, then for $x=100 \Rightarrow y=200$

This is particularly useful in scientific contexts.

## Stability

Symbolic regression is high variance—small dataset changes may produce different expressions:

**Example**:
- Day 1: $y = 3x_1 + 2x_2$
- Day 2: $y = 2.8x_1 + 2.1x_2 + 0.1x_1^2$

Even when the underlying relationship is unchanged.

### Updating Formulas

Updating is difficult because:
- Formulas are globally optimized
- Adding data may change the entire structure

Typical strategies:
- Retrain periodically
- Warm-start search with previous formulas

## Error Behaviour

Symbolic regression optimizes a trade-off:

$$\text{error} + \text{formula complexity}$$

Therefore, symbolic models may sacrifice some predictive accuracy in exchange for simplicity. The goal is to discover the underlying generative equation.

## Function Representation

Produces a continuous mathematical relationship:

**Example**: $f(x) = x_1^2 + \sin(x_2)$

**Properties**:
- Smooth
- Differentiable (often)
- Compact

## Interpretability

Interpretability arises directly from the discovered formula.

**Example**: $y = ax + b$

You can directly infer:
- Variable relationships
- Scaling behaviour
- Nonlinear mechanisms

This makes symbolic regression valuable for scientific discovery.