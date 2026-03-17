A Note containing all the information about symbolic regression within this folder:

Symbolic regression
Symbolic regression attempts to discover an explicit functional form
$$
y = f(x)
$$
Example result:
$$
y = 2.1x_1 + \log(x_2) - 0.5x_3^2
$$
The output is a closed-form mathematical expression.
The model is typically represented as an expression tree composed of:
variables
constants
operators ($+$, $-$, $\times$, $\div$)
nonlinear functions ($\log$, $\exp$, $\sin$, etc.)

1. Computational cost
Symbolic regression
Very expensive.
Reasons:
search over expression space
genetic programming
large candidate population.
Training time often scales poorly with feature count.

1. Extrapolation behaviour
Symbolic regression
Because the model is a mathematical function, it can extrapolate.
Example:
If the model learns
$$
y = 2x
$$
then prediction for unseen range:
$$
x=100 \Rightarrow y=200
$$
This is useful in scientific contexts.

1. Stability
Symbolic regression
Symbolic regression is high variance.
Small dataset changes may produce different expressions.
Example:
Day 1
$$
y = 3x_1 + 2x_2
$$
Day 2
$$
y = 2.8x_1 + 2.1x_2 + 0.1x_1^2
$$
Even when the underlying relationship is unchanged.

Symbolic regression
Updating is difficult because:
formulas are globally optimized
adding data may change the entire structure.
Typical strategy:
retrain periodically
warm-start search with previous formulas.

1. Error behaviour
Symbolic regression optimizes a trade-off:
$$
\text{error} + \text{formula complexity}
$$
Therefore symbolic models may sacrifice some predictive accuracy in exchange for simplicity.

Symbolic regression attempts to discover the underlying generative equation.

1. Function representation
Symbolic regression
Produces a continuous mathematical relationship.
Example:
$$
f(x) = x_1^2 + \sin(x_2)
$$
Properties:
smooth
differentiable (often)
compact.

1. Interpretability
Symbolic regression
Interpretability arises from the formula itself.
Example:
$$
y = ax + b
$$
You can directly infer:
variable relationships
scaling behaviour
nonlinear mechanisms.
This is often used for scientific discovery.