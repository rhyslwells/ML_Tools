# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
# ---

# %% [markdown]
# # Example Notebook (Markdown → ipynb)
# s sssssssssssssssssssssssss tttt
# This is a notebook written in Markdown using Jupytext.
#
# ## 📌 Cell 1: Imports

# %%
import math
import matplotlib.pyplot as plt
import packaging

# %% [markdown]
# ## 📌 Cell 2: A Simple Calculation

# %%
x = 3
area = math.pi * x ** 2
area

# %% [markdown]
# ## 📌 Cell 3: Plotting

# %%
xs = range(10)
ys = [i**2 for i in xs]

plt.plot(xs, ys)
plt.title("y = x^2")
plt.show()
