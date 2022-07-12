"""
Global config module added to resolve some issues, like the implicit default opset choice.

Policy: when a new opset version is deemed stable enough, Steelix will bump the default here.
This choice may be affected with ``config.set_default_opset(op)`` with an opset module as the parameter.

Note that Steelix internal may use the default opset - often to use the Identity operator for renaming.
Additionally, Arrow uses it for operator overloading.
"""

import sys

MISSING = object()
_default_opset = MISSING


def get_default_opset():
    """Access the default opset module. It is loaded in lazily (unless it was already set by ``set_default_opset``)."""
    global _default_opset
    if _default_opset is MISSING:
        try:
            import steelix.opset.ai.onnx.v17 as op
        except ImportError as e:
            print(
                f"Failed to import default_opset (ai.onnx@17): {str(e)}\nIs there a cyclic dependency?",
                file=sys.stderr,
            )
            op = None
        _default_opset = op
    return _default_opset


def set_default_opset(op):
    """Override the default global opset module from the 'stable' choice of Steelix."""
    global _default_opset
    _default_opset = op


__all__ = ["get_default_opset", "set_default_opset"]
