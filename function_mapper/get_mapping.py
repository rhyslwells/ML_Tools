import ast
import os
import json
from collections import defaultdict

ROOT_PATH = r"C:\Users\RhysL\Desktop\Projects\Data-Archive-Explorer\src"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "output.json")

# Folders to ignore (by name, anywhere in the tree)
EXCLUDE_DIRS = {
    "__pycache__",
    "data_archive_explorer.egg-info",
    "lib",
}


class FunctionCollector(ast.NodeVisitor):
    def __init__(self, rel_path):
        self.rel_path = rel_path
        self.functions = {}
        self.current_function = None

    def visit_FunctionDef(self, node):
        func_id = f"{self.rel_path}::{node.name}"
        docstring = ast.get_docstring(node)

        self.functions[func_id] = {
            "name": node.name,
            "file": self.rel_path,
            "docstring": docstring,
            "calls": []
        }

        prev = self.current_function
        self.current_function = func_id

        self.generic_visit(node)

        self.current_function = prev

    def visit_Call(self, node):
        if self.current_function is None:
            return

        call_name = self._get_call_name(node)
        if call_name:
            self.functions[self.current_function]["calls"].append(call_name)

        self.generic_visit(node)

    def _get_call_name(self, node):
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None


def get_all_py_files(root_path):
    py_files = []

    for root, dirs, files in os.walk(root_path):
        # In-place filter of directories (prevents traversal into them)
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, root_path)
                py_files.append((full_path, rel_path))

    return py_files


def first_pass_collect(root_path):
    all_functions = {}

    for full_path, rel_path in get_all_py_files(root_path):
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except Exception:
            continue

        collector = FunctionCollector(rel_path)
        collector.visit(tree)

        all_functions.update(collector.functions)

    return all_functions


def build_name_index(functions):
    index = defaultdict(list)

    for func_id, meta in functions.items():
        index[meta["name"]].append(func_id)

    return index


def resolve_calls(functions, name_index):
    for func_id, meta in functions.items():
        raw_calls = meta["calls"]
        resolved = []

        for call_name in raw_calls:
            matches = name_index.get(call_name, [])

            if len(matches) == 1:
                resolved.append(matches[0])
            elif len(matches) > 1:
                resolved.extend(matches)

        # deduplicate
        seen = set()
        meta["calls"] = [x for x in resolved if not (x in seen or seen.add(x))]


def main():
    print("Collecting functions...")
    functions = first_pass_collect(ROOT_PATH)

    print(f"Total functions found: {len(functions)}")

    print("Building name index...")
    name_index = build_name_index(functions)

    print("Resolving calls...")
    resolve_calls(functions, name_index)

    print("Writing output...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(functions, f, indent=2)

    print(f"Done. Output written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()