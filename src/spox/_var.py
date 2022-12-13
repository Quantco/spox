import typing
from typing import Any, Optional, Union

import numpy

from . import _type_system
from ._config import get_default_opset
from ._shape import Shape

if typing.TYPE_CHECKING:
    from ._node import Node


class Var:
    """
    Abstraction for a single ONNX value, like a tensor, that can be passed around.
    It depends on a given output (``which``) of an operator ``op`` (represented in a ``Node``).

    The ``type`` is inferred and checked. If it is ``None``, it is unknown and should pass all type checks.

    The ``value`` field may be propagated in case a ``Var`` always has a constant value.
    This is useful for type & shape inference.

    The state of a ``Var`` should not be modified, and only the ``type`` and ``value`` fields should be accessed.
    (as ``_op`` and ``_which`` are primarily stored for building a Graph).

    Should not be constructed directly - the main source of Vars is operator constructors & ``arguments``-like.
    """

    type: Optional[_type_system.Type]
    _value: Optional[Any]
    _op: "Node"
    _name: Optional[str]

    def __init__(
        self,
        op: "Node",
        type_: Optional[_type_system.Type],
        value: Optional[Any] = None,
    ):
        self.type = type_
        self._value = value
        self._op = op
        self._name = None
        if not self._value_matches_type(value, type_):
            raise TypeError(
                f"Propagated value {value} of type {type(value)} does not match expected type {type_}."
            )

    def _rename(self, name: Optional[str]):
        """Mutates the internal state of the Var, overriding its name as given."""
        self._name = name

    @staticmethod
    def _value_matches_type(
        value: Optional[Any], type_: Optional[_type_system.Type]
    ) -> bool:
        if value is None or type_ is None:
            return True
        if isinstance(type_, _type_system.Tensor):
            return (
                isinstance(value, numpy.ndarray)
                and value.dtype.type is type_.dtype.type
                and Shape.from_simple(value.shape) <= type_._shape
            )
        elif isinstance(type_, _type_system.Optional):
            return value is Nothing or Var._value_matches_type(value, type_.elem_type)
        elif isinstance(type_, _type_system.Sequence):
            return isinstance(value, list) and all(
                Var._value_matches_type(elem, type_.elem_type) for elem in value
            )
        return True

    @property
    def _which_output(self) -> Optional[str]:
        """Return the name of the output field that this var is stored in under ``self._op``."""
        op_outs = self._op.outputs.as_dict()
        candidates = [key for key, var in op_outs.items() if var is self]
        return candidates[0] if candidates else None

    def __repr__(self) -> str:
        nm = repr(self._name) + " " if self._name is not None else ""
        op_repr = self._op.get_op_repr() if self._op else "??"
        which = self._which_output
        is_unary = len(self._op.outputs.as_dict()) <= 1
        which_repr = "->??" if which is None else (f"->{which}" if is_unary else "")
        return (
            f"<Var {nm}from {op_repr}{which_repr} of {self.type}"
            f"{'' if self._value is None else ' = ' + str(self._value)}>"
        )

    def unwrap_type(self) -> _type_system.Type:
        """
        Returns
        -------
        _type_system.Type
            The type of the Var.
        Raises
        ------
        TypeError
            If ``type is None`` (the type of this ``Var`` is unknown).
        """
        if self.type is None:
            raise TypeError("Cannot unwrap requested type for Var, as it is unknown.")
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

    @property
    def shape(self) -> _type_system.SimpleShape:
        """Equivalent to ``self.unwrap_tensor().shape``."""
        return self.unwrap_tensor().shape

    @property
    def dtype(self) -> _type_system.SimpleShape:
        """Equivalent to ``self.unwrap_tensor().dtype``."""
        return self.unwrap_tensor().shape

    def __add__(self, other) -> "Var":
        if isinstance(other, Var):
            return get_default_opset().add(self, other)
        return NotImplemented

    def __sub__(self, other) -> "Var":
        if isinstance(other, Var):
            return get_default_opset().sub(self, other)
        return NotImplemented

    def __mul__(self, other) -> "Var":
        if isinstance(other, Var):
            return get_default_opset().mul(self, other)
        return NotImplemented

    def __truediv__(self, other) -> "Var":
        if isinstance(other, Var):
            return get_default_opset().div(self, other)
        return NotImplemented

    def __and__(self, other) -> "Var":
        if isinstance(other, Var):
            return get_default_opset().and_(self, other)
        return NotImplemented

    def __or__(self, other) -> "Var":
        if isinstance(other, Var):
            return get_default_opset().or_(self, other)
        return NotImplemented

    def __xor__(self, other) -> "Var":
        if isinstance(other, Var):
            return get_default_opset().xor(self, other)
        return NotImplemented

    def __invert__(self) -> "Var":
        return get_default_opset().not_(self)


class _NilVar(Var):
    """
    Singleton Var which indicates lack of a value.

    This is used as internally some operator inputs may be None, and it is convenient for the rest of the code
    to actually access Vars instead of special-casing None in every instance.

    Operator inputs/outputs that are unspecified in the ONNX representation receive an empty string as the name of the
    value. This is what a NilVar gets converted to.
    """

    def __init__(self):  # noqa
        self.type = None
        self._value = None
        self._name = ""

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other) -> bool:
        return isinstance(other, _NilVar)

    def __repr__(self) -> str:
        return "<nil>"

    def __bool__(self):
        return False


# Singleton instance for _NilVar.
_nil = _NilVar()
del _NilVar.__init__


def result_type(
    *types: Union[Var, numpy.generic, int, float]
) -> typing.Type[numpy.generic]:
    """Promote type for all given element types/values using ``np.result_type``."""
    return numpy.dtype(
        numpy.result_type(
            *(
                typ.unwrap_tensor().dtype if isinstance(typ, Var) else typ
                for typ in types
            )
        )
    ).type


class _NothingType:
    """Singleton class representing a ``Var``'s value which is optional and missing - rather than lack of value."""

    def __repr__(self) -> str:
        return "Nothing"


Nothing = _NothingType()
