"""Module implementing the main public interface functions in Spox."""

import contextlib
from typing import Dict, Optional, Protocol

import numpy as np
import onnx
from onnx.numpy_helper import to_array

from . import _internal_op
from ._attributes import AttrType
from ._graph import results
from ._inline import _Inline
from ._type_system import Type
from ._var import Var


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
        An unnamed argument variable of given type that may be used as a model input to build a graph.
    """
    return _internal_op.Argument(
        _internal_op.Argument.Attributes(type=AttrType(typ), default=None)
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
    onnx.ModelProto
        An ONNX ModelProto containing operators necessary to compute ``outputs`` from ``inputs``.

        If multiple versions of the ``ai.onnx`` domain are present, the nodes are all converted to the newest one.

        The returned model may be mutated after building to add metadata, docstrings, etc.
    """
    with _temporary_renames(**inputs):
        graph = results(**outputs)
        graph = graph.with_arguments(*inputs.values())
        return graph.to_onnx_model()


class _InlineCall(Protocol):
    """
    A callable returned by ``inline``, taking positional and keyword arguments of type ``Var``,
    and returning a dictionary of names (str) into ``Var``.
    """

    def __call__(self, *args: Var, **kwargs: Var) -> Dict[str, Var]:
        """
        Parameters
        ----------
        args
            Variables passed as model inputs - positional, as they are listed in the model.
        kwargs
            Further variables passed as model inputs - keyword, as they are named in the model.
        Returns
        -------
        Dict[str, Var]
            Variables representing the inlined model's outputs.
        """


def inline(model: onnx.ModelProto) -> _InlineCall:
    """
    Inline an existing ONNX model. Takes and produces ``Var``s.

    Any valid model should be successfully inlined, and the model is assumed to have been checked by ONNX.

    The behaviour of a model is replicated, but some of its metadata (docstring, annotations) may be stripped.

    The opset imports of the target model are significant and the model itself may be adapted
    if its version is inconsistent.

    At build time, an inlined model puts its nodes at the assigned insertion point in the topological ordering.
    Prefixing is applied (with the build system name) to attempt to avoid collisions.
    Build behaviour should be treated as an implementation detail and may change.

    ``inline`` is intended to help achieve:

    - Applying ONNX which may not be produced by Spox directly, like custom operators.
    - Interfacing with other ONNX libraries - for instance converters - and composing existing ONNX models.

    Parameters
    ----------
    model
        Target model to inline.

    Returns
    -------
    _InlineCall
        A callable which takes ``Var`` arguments and returns a dictionary of output names into ``Var``.

        Positional arguments are assigned based on the order they are listed in the model.
        Keyword arguments are assigned based on their names in the model.

        Unspecified arguments are replaced by an initializer of the same name in the model, if one exists.

        Input types are expected to be compatible with the model's graph input types.
        Output types produced are copied from the model's graph output types.

    Raises
    ------
    TypeError
        If the arguments to the callback are supplied incorrectly or the variables are of the wrong type.
    """
    in_names = [i.name for i in model.graph.input]
    in_defaults = {i.name: i for i in model.graph.initializer}
    out_names = [o.name for o in model.graph.output]
    _defaults_msg = f" (defaults {list(in_defaults.keys())})"
    _signature_msg = f"signature {in_names}{_defaults_msg} -> {out_names}"

    def inline_inner(*args: Var, **kwargs: Var) -> Dict[str, Var]:
        for name, arg in zip(in_names, args):
            if name in kwargs:
                raise TypeError(
                    f"inline callback got multiple values for argument '{name}', {_signature_msg}."
                )
            kwargs[name] = arg
        if not (missing := set(in_names) - set(kwargs)) <= set(in_defaults):
            raise TypeError(
                f"inline callback missing required arguments: {missing}, {_signature_msg}."
            )
        for name in missing:
            array = to_array(in_defaults[name])
            if array.dtype == np.dtype(object):
                array = array.astype(str)
            kwargs[name] = _internal_op.constant(array)

        if set(kwargs) != set(in_names):
            raise TypeError(
                f"Error processing arguments, got {set(kwargs)}, expected {set(in_names)}."
            )
        node = _Inline(
            inputs=_Inline.Inputs([kwargs[name] for name in in_names]),
            out_variadic=len(model.graph.output),
            model=model,
        )
        return dict(zip(out_names, node.outputs.outputs))

    return inline_inner


__all__ = ["argument", "build", "inline"]
