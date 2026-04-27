## Objective

Create a Python script that analyses a directory of Python files and produces a function-level call graph in JSON format. The output should integrate directly with NetworkX for querying and visualisation.

---

## Input

* `root_path: str` — root directory containing `.py` files

---

## Output

A single JSON object where:

* Each key is a **unique function identifier**
* Functions with the same name in different files are treated as **distinct nodes**

---

## Function Identifier

$$
\text{id} = \text{relative_file_path} + "::" + \text{function_name}
$$

Example:

```text
utils/math.py::compute_sum
```

---

## JSON Schema

```json
{
  "utils/math.py::compute_sum": {
    "name": "compute_sum",
    "file": "utils/math.py",
    "docstring": "Compute sum of values",
    "calls": ["utils/math.py::validate_input"]
  }
}
```

---

## Core Behaviour

### 1. File Traversal

* Recursively scan `root_path`
* Include only `.py` files
* Store **relative paths** (important for stable IDs)

---

### 2. Function Extraction

Using `ast`:

* Extract all `FunctionDef`
* For each function:

  * name
  * file path
  * docstring

---

### 3. Call Extraction

Within each function:

* Traverse `ast.Call`
* Extract call names from:

  * `ast.Name` → `func`
  * `ast.Attribute` → `obj.func` → store `"func"`

Store these as **raw call names initially**

---

### 4. Resolution Step (Required)

Construct:

$$
S = {\text{all } (file, function_name)}
$$

Then for each function:

* For each `call_name`:

  * Find all matches in $S$ where:
    $$
    \text{function_name} = \text{call_name}
    $$

* If exactly one match → include it

* If multiple matches → include **all matches**

* If no match → ignore (external function)

---

### Resolution Rule

This preserves your design choice:

> Same function names across files remain separate nodes

Example:

```text
a.py::process
b.py::process
```

A call to `"process"` becomes:

```json
"calls": ["a.py::process", "b.py::process"]
```

---

## Final Output Rule

* `"calls"` contains only **resolved internal function IDs**
* No external library calls included

---

## Example

```json
{
  "a.py::f1": {
    "name": "f1",
    "file": "a.py",
    "docstring": null,
    "calls": ["a.py::f2", "b.py::f2"]
  },
  "a.py::f2": {
    "name": "f2",
    "file": "a.py",
    "docstring": null,
    "calls": []
  },
  "b.py::f2": {
    "name": "f2",
    "file": "b.py",
    "docstring": null,
    "calls": []
  }
}
```

---

## NetworkX Usage

### Build Graph

```python
import json
import networkx as nx

with open("output.json") as f:
    data = json.load(f)

G = nx.DiGraph()

for func_id, meta in data.items():
    G.add_node(func_id, **meta)
    for callee in meta["calls"]:
        G.add_edge(func_id, callee)
```

---

## Queries

### Downstream dependencies

```python
nx.descendants(G, "a.py::f1")
```

---

### Upstream dependencies

```python
nx.ancestors(G, "a.py::f1")
```

---

### Dead code detection

```python
[n for n in G.nodes if G.in_degree(n) == 0]
```

---

### Path tracing

```python
nx.shortest_path(G, "a.py::f1", "b.py::f2")
```

---

## Visualisation

```python
import matplotlib.pyplot as plt

nx.draw(G, with_labels=True, node_size=500, font_size=8)
plt.show()
```

---

## Constraints

* No import resolution
* No class method support
* No dynamic call handling
* Only static, direct calls

---

## Design Outcome

This gives you:

* A clean directed graph:
  $$
  G = (V, E)
  $$

* Nodes = functions

* Edges = call relationships

* Fully compatible with:

  * NetworkX analysis
  * Simple visualisation
  * Notebook-based exploration

