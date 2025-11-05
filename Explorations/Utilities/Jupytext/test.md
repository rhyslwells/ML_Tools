---
jupytext:
  formats: md,ipynb
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Example Notebook (Markdown → ipynb)

This is a notebook written in Markdown using Jupytext.

## 📌 Cell 1: Imports

```python
import math
import matplotlib.pyplot as plt
```

## 📌 Cell 2: A Simple Calculation

```python
x = 3
area = math.pi * x ** 2
area
```

## 📌 Cell 3: Plotting

```python
xs = range(10)
ys = [i**2 for i in xs]

plt.plot(xs, ys)
plt.title("y = x^2")
plt.show()
```
