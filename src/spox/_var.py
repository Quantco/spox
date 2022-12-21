import typing
from typing import Optional, Union

import numpy

from . import _type_system, _value_prop
from ._config import get_default_opset

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
    _value: Optional[_value_prop.PropValue]
    _op: "Node"
    _name: Optional[str]

    def __init__(
        self,
        op: "Node",
        type_: Optional[_type_system.Type],
        value: Optional[_value_prop.PropValue] = None,
    ):
        if type_ is not None and not isinstance(type_, _type_system.Type):
            raise TypeError("The type field of a Var must be a Spox Type.")
        if value is not None and not isinstance(value, _value_prop.PropValue):
            raise TypeError("The propagated value field of a Var must be a PropValue.")
        if value is not None and value.type != type_:
            raise ValueError(
                f"The propagated value type ({value.type}) and actual Var type ({type_}) must be the same."
            )

        self.type = type_
        self._value = value
        self._op = op
        self._name = None

    def _rename(self, name: Optional[str]):
        """Mutates the internal state of the Var, overriding its name as given."""
        self._name = name

    @property
    def _which_output(self) -> Optional[str]:
        """Return the name of the output field that this var is stored in under ``self._op``."""
        if self._op is None:
            return None
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
            f"{'' if self._value is None else ' = ' + str(self._value.to_ort_value())}>"
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
