## 🔧 Basic Commands

### Pair notebook with script

```bash
jupytext --set-formats ipynb,py:percent notebook.ipynb
```

### Sync paired files

```bash
jupytext --sync notebook.ipynb
jupytext --set-formats ipynb,py:percent test.ipynb
```

### Convert between formats
These do create a new file.

```bash
jupytext notebook.ipynb --to py:percent
jupytext script.py --to notebook
```

---

## 🧾 Format Options

| Format       | Description                      | VS Code Compatible |
| ------------ | -------------------------------- | ------------------ |
| `py:percent` | `# %%` cell markers              | ✅ Yes              |


## 🧠 Tips

* Use `# %%` format for `.py` files so VS Code treats them as notebooks.
* Run `jupytext --sync` after editing either file.
* Commit `.py` to Git, regenerate `.ipynb` as needed.
