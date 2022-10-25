import typing
from typing import Any, Generic, Optional, TypeVar, Union

import numpy

from . import type_system
from .shape import Shape
from .type_system import Tensor, Type

if typing.TYPE_CHECKING:
    from .node import Node

T = TypeVar("T")


class Arrow(Generic[T]):
    """
    Abstraction for a single ONNX value, like a tensor, that can be passed around.
    It depends on a given output (``which``) of an operator ``op`` (represented in a ``Node``).

    The ``type`` is inferred and checked. If it is ``None``, it is unknown and should pass all type checks.

    The ``value`` field may be propagated in case an Arrow always has a constant value.
    This is useful for type & shape inference.

    The state of an Arrow should not be modified, and only the ``type`` and ``value`` fields should be accessed.
    (as ``_op`` and ``_which`` are primarily stored for building a Graph).

    Should not be constructed directly - the main source of Arrows is operator constructors & ``arguments``-like.
    """

    type: Optional[Type]
    value: Optional[Any]
    _op: "Node"
    _name: Optional[str]

    def __init__(self, op: "Node", type_: Optional[Type], value: Optional[Any] = None):
        self.type = type_
        self.value = value
        self._op = op
        self._name = None
        if not self._value_matches_type(value, type_):
            raise TypeError(
                f"Propagated value {value} of type {type(value)} does not match expected type {type_}."
            )

    def _rename(self, name: Optional[str]):
        """Mutates the internal state of the Arrow, overriding its name as given."""
        self._name = name

    @staticmethod
    def _value_matches_type(value: Optional[Any], type_: Optional[Type]) -> bool:
        if value is None or type_ is None:
            return True
        if isinstance(type_, Tensor):
            return (
                isinstance(value, numpy.ndarray)
                and value.dtype.type is type_.elem_type
                and Shape.from_simple(value.shape) <= type_.shape
            )
        elif isinstance(type_, type_system.Optional):
            return value is Nothing or Arrow._value_matches_type(value, type_.elem_type)
        elif isinstance(type_, type_system.Sequence):
            return isinstance(value, list) and all(
                Arrow._value_matches_type(elem, type_.elem_type) for elem in value
            )
        return True

    @property
    def default_opset(self):
        """Default operator set used for operator overloading."""
        from . import config

        return config.get_default_opset()

    @property
    def _which_output(self) -> Optional[str]:
        """Return the name of the output field that this arrow is stored in under ``self._op``."""
        op_outs = self._op.outputs.as_dict()
        candidates = [key for key, arrow in op_outs.items() if arrow is self]
        return candidates[0] if candidates else None

    def __repr__(self) -> str:
        nm = repr(self._name) + " " if self._name is not None else ""
        op_repr = self._op.get_op_repr() if self._op else "??"
        which = self._which_output
        is_unary = len(self._op.outputs.as_dict()) <= 1
        which_repr = "->??" if which is None else (f"->{which}" if is_unary else "")
        return (
            f"<Arrow {nm}from {op_repr}{which_repr} of {self.type}"
            f"{'' if self.value is None else ' = ' + str(self.value)}>"
        )

    def unwrap_type(self) -> Type:
        """
        Convenience function that raises if the type is unknown and returns it otherwise.

        Raises
        -------
        TypeError
            If ``type`` is None.
        """
        if self.type is None:
            raise TypeError("Cannot unwrap requested type for Arrow, as it is unknown.")
        return self.type

    def unwrap_tensor(self) -> Tensor:
        """
        Convenience function that raises if the type is not a Tensor and returns one otherwise.

        Raises
        -------
        TypeError
            If ``type`` is not a Tensor.
        """
        return self.unwrap_type().unwrap_tensor()

    def __add__(self, other) -> "Arrow":
        if isinstance(other, Arrow):
            return self.default_opset.add(self, other)
        return NotImplemented

    def __sub__(self, other) -> "Arrow":
        if isinstance(other, Arrow):
            return self.default_opset.sub(self, other)
        return NotImplemented

    def __mul__(self, other) -> "Arrow":
        if isinstance(other, Arrow):
            return self.default_opset.mul(self, other)
        return NotImplemented

    def __truediv__(self, other) -> "Arrow":
        if isinstance(other, Arrow):
            return self.default_opset.div(self, other)
        return NotImplemented

    def __and__(self, other) -> "Arrow":
        if isinstance(other, Arrow):
            return self.default_opset.and_(self, other)
        return NotImplemented

    def __or__(self, other) -> "Arrow":
        if isinstance(other, Arrow):
            return self.default_opset.or_(self, other)
        return NotImplemented

    def __xor__(self, other) -> "Arrow":
        if isinstance(other, Arrow):
            return self.default_opset.xor(self, other)
        return NotImplemented

    def __invert__(self) -> "Arrow":
        return self.default_opset.not_(self)


class _NilArrow(Arrow):
    """
    Singleton Arrow which indicates lack of a value.

    This is used as internally some operator inputs may be None, and it is convenient for the rest of the code
    to actually access Arrows instead of special-casing None in every instance.

    Operator inputs/outputs that are unspecified in the ONNX representation receive an empty string as the name of the
    value. This is what a NilArrow gets converted to.
    """

    def __init__(self):  # noqa
        self.type = None
        self.value = None
        self._name = ""

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other) -> bool:
        return isinstance(other, _NilArrow)

    def __repr__(self) -> str:
        return "<nil>"

    def __bool__(self):
        return False


# Singleton instance for _NilArrow.
_nil = _NilArrow()
del _NilArrow.__init__


def result_type(
    *types: Union[Arrow, numpy.generic, int, float]
) -> typing.Type[numpy.generic]:
    """Promote type for all given element types/values using ``np.result_type``."""
    return numpy.dtype(
        numpy.result_type(
            *(
                typ.unwrap_tensor().elem_type if isinstance(typ, Arrow) else typ
                for typ in types
            )
        )
    ).type


class _NothingType:
    """Singleton class representing an Arrow's value which is optional and missing - rather than lack of value."""

    def __repr__(self) -> str:
        return "Nothing"


Nothing = _NothingType()
