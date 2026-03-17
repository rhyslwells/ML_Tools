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