"""Static check of LLM-written Python before it runs.

This is a first line of defence, not a sandbox: it rejects code that imports modules outside an
allow-list, touches files, the network or the interpreter, or reaches for dunder attributes (the
usual way out of restricted Python). The child process limits in python_exec still apply, and a
public deployment can switch run_python off entirely (ALLOW_PYTHON=false).
"""
from __future__ import annotations

import ast

ALLOWED_MODULES = {
    "pandas", "numpy", "scipy", "statsmodels", "sklearn", "math", "statistics", "json", "re",
    "datetime", "collections", "itertools", "functools", "decimal", "fractions", "random", "string",
    "textwrap", "operator", "dataclasses", "typing",
}
BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "memoryview", "exit", "quit", "help",
}
# pandas/numpy methods that read or write files, or run shell commands
BLOCKED_ATTRS = {
    "to_csv", "to_excel", "to_parquet", "to_pickle", "to_json", "to_sql", "to_html", "to_feather",
    "to_hdf", "to_clipboard", "to_latex", "to_xml", "to_stata", "to_orc", "tofile", "save", "savez",
    "savetxt", "load", "loadtxt", "fromfile", "genfromtxt", "system", "popen",
}


class UnsafeCodeError(ValueError):
    pass


def check_code(code: str) -> None:
    """Raise UnsafeCodeError with a plain reason if the code is not allowed."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise UnsafeCodeError(f"Syntax error on line {e.lineno}: {e.msg}") from e
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check_module(alias.name)
        elif isinstance(node, ast.ImportFrom):
            _check_module(node.module or "")
        elif isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            raise UnsafeCodeError(f"'{node.id}' is not allowed in analysis code.")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise UnsafeCodeError("Double-underscore attributes are not allowed in analysis code.")
            if node.attr in BLOCKED_ATTRS or node.attr.startswith("read_"):
                raise UnsafeCodeError(f"'.{node.attr}' reads or writes files, which analysis code can't do. "
                                      "The data is already loaded as DataFrames.")


def _check_module(name: str) -> None:
    root = name.split(".")[0]
    if root not in ALLOWED_MODULES:
        raise UnsafeCodeError(f"Importing '{name}' is not allowed. Use pandas, numpy, scipy, statsmodels "
                              "or sklearn; the data is already loaded.")
