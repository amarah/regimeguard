"""Dashboard caches must vary with inputs and eventually refresh market data."""
import ast
from pathlib import Path


APP = Path(__file__).parents[1] / "app.py"


def cached_functions():
    tree = ast.parse(APP.read_text())
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("cached_")
    }


def test_position_cache_includes_every_analysis_input():
    function = cached_functions()["cached_sizes"]

    assert [argument.arg for argument in function.args.args] == [
        "tickers", "capital", "mult"
    ]
    assert all(not argument.arg.startswith("_") for argument in function.args.args)


def test_dashboard_caches_expire_after_twelve_hours():
    for function in cached_functions().values():
        decorator = next(
            item for item in function.decorator_list
            if isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and item.func.attr == "cache_data"
        )
        ttl = next(keyword.value for keyword in decorator.keywords
                   if keyword.arg == "ttl")
        assert ast.literal_eval(ttl.left) * ast.literal_eval(ttl.right) == 12 * 3600
