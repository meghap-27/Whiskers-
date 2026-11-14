"""Executed in a child process by python_exec.run_python. Reads a JSON spec on stdin."""
import ast
import contextlib
import io
import json
import sys
import traceback

MARK = "__AGENT_RESULT__"


def _limits(spec):
    try:
        import resource
        mem = spec["memory_mb"] * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_CPU, (spec["cpu_s"], spec["cpu_s"]))
    except Exception:
        pass  # not available on Windows


def main():
    spec = json.loads(sys.stdin.read())
    import numpy as np
    import pandas as pd
    _limits(spec)  # after heavy imports so they are not counted against the cap

    ns = {"pd": pd, "np": np}
    try:
        import scipy.stats as stats
        ns["stats"] = stats
    except ImportError:
        pass
    for name, path in spec["datasets"].items():
        ns[name] = pd.read_parquet(path)
    if spec.get("main"):
        ns["df"] = ns[spec["main"]]
    if spec.get("result"):
        ns["result"] = pd.read_parquet(spec["result"])

    buf = io.StringIO()
    out = {"ok": True, "stdout": "", "last_value": None, "has_output": False}
    try:
        tree = ast.parse(spec["code"], mode="exec")
        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = ast.Expression(tree.body.pop().value)
        with contextlib.redirect_stdout(buf):
            exec(compile(tree, "<analysis>", "exec"), ns)
            value = eval(compile(last_expr, "<analysis>", "eval"), ns) if last_expr else None
        output = ns.get("output")
        if output is None and isinstance(value, (pd.DataFrame, pd.Series)):
            output = value
        if isinstance(output, pd.Series):
            output = output.reset_index()
        if isinstance(output, pd.DataFrame):
            o = output.reset_index() if not isinstance(output.index, pd.RangeIndex) else output
            o.columns = [str(c) for c in o.columns]
            o.to_parquet(spec["output_path"], index=False)
            out["has_output"] = True
        if isinstance(value, np.generic):
            value = value.item()  # show 6, not np.int64(6)
        if value is not None:
            with pd.option_context("display.max_rows", 50, "display.max_columns", 30, "display.width", 200):
                out["last_value"] = repr(value)[:4000]
    except Exception:
        out["ok"] = False
        out["error"] = traceback.format_exc(limit=3)[-2000:]
    out["stdout"] = buf.getvalue()[-4000:]
    sys.stdout.write("\n" + MARK + json.dumps(out, default=str))


if __name__ == "__main__":
    main()
