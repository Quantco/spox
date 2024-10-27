# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

import typing
from typing import Callable, Optional, TypeVar, Union

import numpy as np

from . import _type_system, _value_prop

if typing.TYPE_CHECKING:
    from ._node import Node

F = TypeVar("F", bound=Callable)


class VarInfo:
    """
    Abstraction for a single ONNX value - like a tensor - that can be passed around in Python code.

    A ``VarInfo`` represents some output of an operator.
    This operator is stored internally to allow reproducing the graph.

    The ``type`` field is inferred and checked by operators.
    It may be ``None`` if type inference failed, in which case it is unknown and should pass all type checks.
    However, untyped ``VarInfo`` objects may not be used in some contexts.
    Keep in mind that the types themselves may have some information missing.
    For instance, tensors allow missing rank and shape information.

    There is an implicit value propagation mechanism, powered by the ONNX reference implementation.
    Values may be propagated if a ``VarInfo`` always has a known and constant value at runtime.
    This is used for type & shape inference. For instance, Reshape to a constant shape can have the shape inferred.

    ``VarInfo`` should be treated as strictly immutable.
    If a ``VarInfo`` or any of its fields are modified, the behaviour is undefined and the produced graph may be invalid.

    Protected fields are to be treated as internal.
    Useful data is also shown by the string representation, but it should be treated as debug information.

    Should not be constructed directly - the main source of ``VarInfo`` objects are operator constructors.
    """

    type: Optional[_type_system.Type]
    _op: "Node"
    _name: Optional[str]

    def __init__(
        self,
        op: "Node",
        type_: Optional[_type_system.Type],
    ):
        """The initializer of ``VarInfo`` is protected. Use operator constructors to construct them instead."""
        if type_ is not None and not isinstance(type_, _type_system.Type):
            raise TypeError("The type field of a VarInfo must be a Spox Type.")

        self.type = type_
        self._op = op
        self._name = None

    def _rename(self, name: Optional[str]):
        """Mutates the internal state of the VarInfo, overriding its name as given."""
        self._name = name

    @property
    def _which_output(self) -> Optional[str]:
        """Return the name of the output field that this var is stored in under ``self._op``."""
        if self._op is None:
            return None
        op_outs = self._op.outputs.get_vars()
        candidates = [key for key, var in op_outs.items() if var is self]
        return candidates[0] if candidates else None

    def __repr__(self) -> str:
        nm = repr(self._name) + " " if self._name is not None else ""
        op_repr = self._op.get_op_repr() if self._op else "??"
        which = self._which_output
        is_unary = len(self._op.outputs) <= 1 if self._op else True
        which_repr = "->??" if which is None else (f"->{which}" if is_unary else "")
        return f"<VarInfo {nm}from {op_repr}{which_repr} of {self.type}>"

    def unwrap_type(self) -> _type_system.Type:
        """
        Return the :class:`~spox.Type` of ``self``, unless it is unknown.

        Returns
        -------
        _type_system.Type
            The type of the VarInfo.

        Raises
        ------
        TypeError
            If ``type is None`` (the type of this ``VarInfo`` is unknown).
        """
        if self.type is None:
            raise TypeError(
                "Cannot unwrap requested type for VarInfo, as it is unknown."
            )
        return self.type

    def unwrap_tensor(self) -> _type_system.Tensor:
        """Equivalent to ``self.unwrap_type().unwrap_tensor()``."""
        return self.unwrap_type().unwrap_tensor()

    def unwrap_sequence(self) -> _type_system.Sequence:
        """Equivalent to ``self.unwrap_type().unwrap_sequence()``."""
        return self.unwrap_type().unwrap_sequence()

    def unwrap_optional(self) -> _type_system.Optional:
        """Equivalent to ``self.unwrap_type().unwrap_optional()``."""
        return self.unwrap_type().unwrap_optional()

    def __copy__(self) -> "VarInfo":
        # Simply return `self` to ensure that "copies" are still equal
        # during the build process
        return self

    def __deepcopy__(self, _) -> "VarInfo":
        raise ValueError("'VarInfo' objects cannot be deepcopied.")


class Var:
    """
    Abstraction for a single ONNX value - like a tensor - that can be passed around in Python code.

    A ``VarInfo`` represents some output of an operator.
    This operator is stored internally to allow reproducing the graph.

    The ``type`` field is inferred and checked by operators.
    It may be ``None`` if type inference failed, in which case it is unknown and should pass all type checks.
    However, untyped ``VarInfo`` objects may not be used in some contexts.
    Keep in mind that the types themselves may have some information missing.
    For instance, tensors allow missing rank and shape information.

    There is an implicit value propagation mechanism, powered by the ONNX reference implementation.
    Values may be propagated if a ``VarInfo`` always has a known and constant value at runtime.
    This is used for type & shape inference. For instance, Reshape to a constant shape can have the shape inferred.

    ``VarInfo`` should be treated as strictly immutable.
    If a ``VarInfo`` or any of its fields are modified, the behaviour is undefined and the produced graph may be invalid.

    Protected fields are to be treated as internal.
    Useful data is also shown by the string representation, but it should be treated as debug information.

    Should not be constructed directly - the main source of ``VarInfo`` objects are operator constructors.
    """

    _var_info: VarInfo
    _value: Optional[_value_prop.PropValue]

    def __init__(
        self,
        var_info: VarInfo,
        value: Optional[_value_prop.PropValue] = None,
    ):
        """The initializer of ``Var`` is protected. Use operator constructors to construct them instead."""
        if value is not None and not isinstance(value, _value_prop.PropValue):
            raise TypeError(
                "The propagated value field of a VarInfo must be a PropValue."
            )
        if value is not None and value.type != var_info._type:
            raise ValueError(
                f"The propagated value type ({value.type}) and actual VarInfo type ({type_}) must be the same."
            )

        self._var_info = var_info
        self._value = value

    def _get_value(self) -> "_value_prop.ORTValue":
        """Get the propagated value in this VarInfo and convert it to the ORT format. Raises if value is missing."""
        if self._value is None:
            raise ValueError("No propagated value associated with this VarInfo.")
        return self._value.to_ort_value()

    def __repr__(self) -> str:
        nm = repr(self._name) + " " if self._name is not None else ""
        op_repr = self._op.get_op_repr() if self._op else "??"
        which = self._which_output
        is_unary = len(self._op.outputs) <= 1 if self._op else True
        which_repr = "->??" if which is None else (f"->{which}" if is_unary else "")
        return (
            f"<Var {nm}from {op_repr}{which_repr} of {self.type}"
            f"{'' if self._value is None else ' = ' + str(self._value.to_ort_value())}>"
        )

    def unwrap_type(self) -> _type_system.Type:
        """
        Return the :class:`~spox.Type` of ``self``, unless it is unknown.

        Returns
        -------
        _type_system.Type
            The type of the VarInfo.

        Raises
        ------
        TypeError
            If ``type is None`` (the type of this ``VarInfo`` is unknown).
        """
        if self.type is None:
            raise TypeError(
                "Cannot unwrap requested type for VarInfo, as it is unknown."
            )
        return self.type

    def unwrap_tensor(self) -> _type_system.Tensor:
        """Equivalent to ``self.unwrap_type().unwrap_tensor()``."""
        return self.unwrap_type().unwrap_tensor()

    def unwrap_sequence(self) -> _type_system.Sequence:
        """Equivalent to ``self.unwrap_type().unwrap_sequence()``."""
        return self.unwrap_type().unwrap_sequence()

    def unwrap_optional(self) -> _type_system.Optional:
        """Equivalent to ``self.unwrap_type().unwrap_optional()``."""
        return self.unwrap_type().unwrap_optional()

    def __copy__(self) -> "VarInfo":
        # Simply return `self` to ensure that "copies" are still equal
        # during the build process
        return self

    def __deepcopy__(self, _) -> "VarInfo":
        raise ValueError("'VarInfo' objects cannot be deepcopied.")


def result_type(
    *types: Union[VarInfo, np.generic, int, float],
) -> type[np.generic]:
    """Promote type for all given element types/values using ``np.result_type``."""
    return np.dtype(
        np.result_type(
            *(
                typ.unwrap_tensor().dtype if isinstance(typ, VarInfo) else typ
                for typ in types
            )
        )
    ).type
