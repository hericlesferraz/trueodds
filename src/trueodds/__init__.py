"""trueodds: a decision model that returns a probability per option."""

__version__ = "0.0.1"


def __getattr__(name: str):
    # `trueodds.load` without importing torch and transformers on `import trueodds`.
    if name == "load":
        from trueodds.infer import load

        return load
    raise AttributeError(f"module 'trueodds' has no attribute {name!r}")
