"""Module implementing the publicly-accessible side of building in Spox."""

import contextlib
from typing import Dict, Optional

import onnx

from ._attributes import AttrType
from ._graph import results
from ._internal_op import Argument as _Argument
from ._type_system import Type
from ._var import Var
from ._varfields import NoVars


def argument(typ: Type) -> Var:
    """
    Create an argument variable which may be used as a model input.

    Parameters
    ----------
    typ
        The type of the created argument variable.
    Returns
    -------
    arg
        An unnamed argument variable of given type that may be used to construct a graph.

        The returned value is an ``Arg``, which is a subtype of ``Var``.
        Only ``Arg``s may be supplied as a model input for ``build``.
    """
    return _Argument(
        _Argument.Attributes(type=AttrType(typ), default=None),
        NoVars(),
    ).outputs.arg


@contextlib.contextmanager
def _temporary_renames(**kwargs: Var):
    # The build code can't really special-case variable names that are not just ``Var._name``.
    # So we set names here and reset them afterwards.
    name: Optional[str]
    pre: Dict[Var, Optional[str]] = {}
    try:
        for name, arg in kwargs.items():
            pre[arg] = arg._name
            arg._rename(name)
        yield
    finally:
        for arg, name in pre.items():
            arg._rename(name)


def build(inputs: Dict[str, Var], outputs: Dict[str, Var]) -> onnx.ModelProto:
    """
    Builds an ONNX Model with given model inputs and outputs.

    Parameters
    ----------
    inputs
        Model inputs. Keys are names, values must be results of ``argument``.
    outputs
        Model outputs. Keys are names, values may be any ``Var``.
        Building will resolve what nodes were used in the construction of output variables.

    Returns
    -------
        An ONNX ModelProto containing operators necessary to compute ``outputs`` from ``inputs``.

        If multiple versions of the ``ai.onnx`` domain are present, the nodes are all converted to the newest one

        The returned model may be mutated after building to add metadata, docstrings, etc.
    """
    with _temporary_renames(**inputs):
        graph = results(**outputs)
        graph = graph.with_arguments(*inputs.values())
        return graph.to_onnx_model()


__all__ = ["argument", "build"]
