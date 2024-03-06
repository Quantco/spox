import typing
from typing import Any, Callable, ClassVar, Optional, TypeVar, Union

import numpy

from . import _type_system, _value_prop

if typing.TYPE_CHECKING:
    from ._node import Node

F = TypeVar("F", bound=Callable)


class NotImplementedOperatorDispatcher:
    def _not_impl(self, *_):
        return NotImplemented

    add = sub = mul = truediv = floordiv = neg = and_ = or_ = xor = not_ = _not_impl


class Var:
    """
    Abstraction for a single ONNX value - like a tensor - that can be passed around in Python code.

    A ``Var`` represents some output of an operator.
    This operator is stored internally to allow reproducing the graph.

    The ``type`` field is inferred and checked by operators.
    It may be ``None`` if type inference failed, in which case it is unknown and should pass all type checks.
    However, untyped ``Var`` objects may not be used in some contexts.
    Keep in mind that the types themselves may have some information missing.
    For instance, tensors allow missing rank and shape information.

    There is an implicit value propagation mechanism, powered by the ONNX reference implementation.
    Values may be propagated if a ``Var`` always has a known and constant value at runtime.
    This is used for type & shape inference. For instance, Reshape to a constant shape can have the shape inferred.

    ``Var`` should be treated as strictly immutable.
    If a ``Var`` or any of its fields are modified, the behaviour is undefined and the produced graph may be invalid.

    Protected fields are to be treated as internal.
    Useful data is also shown by the string representation, but it should be treated as debug information.

    Should not be constructed directly - the main source of ``Var`` objects are operator constructors.
    """

    type: Optional[_type_system.Type]
    _value: Optional[_value_prop.PropValue]
    _op: "Node"
    _name: Optional[str]

    _operator_dispatcher: ClassVar[Any] = NotImplementedOperatorDispatcher()

    def __init__(
        self,
        op: "Node",
        type_: Optional[_type_system.Type],
        value: Optional[_value_prop.PropValue] = None,
    ):
        """The initializer of ``Var`` is protected. Use operator constructors to construct them instead."""
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
        op_outs = self._op.outputs.get_vars()
        candidates = [key for key, var in op_outs.items() if var is self]
        return candidates[0] if candidates else None

    def _get_value(self) -> "_value_prop.ORTValue":
        """Get the propagated value in this Var and convert it to the ORT format. Raises if value is missing."""
        if self._value is None:
            raise ValueError("No propagated value associated with this Var.")
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

    def __copy__(self) -> "Var":
        # Simply return `self` to ensure that "copies" are still equal
        # during the build process
        return self

    def __deepcopy__(self, _) -> "Var":
        raise ValueError("'Var' objects cannot be deepcopied.")

    def __add__(self, other) -> "Var":
        return Var._operator_dispatcher.add(self, other)

    def __sub__(self, other) -> "Var":
        return Var._operator_dispatcher.sub(self, other)

    def __mul__(self, other) -> "Var":
        return Var._operator_dispatcher.mul(self, other)

    def __truediv__(self, other) -> "Var":
        return Var._operator_dispatcher.truediv(self, other)

    def __floordiv__(self, other) -> "Var":
        return Var._operator_dispatcher.floordiv(self, other)

    def __neg__(self) -> "Var":
        return Var._operator_dispatcher.neg(self)

    def __and__(self, other) -> "Var":
        return Var._operator_dispatcher.and_(self, other)

    def __or__(self, other) -> "Var":
        return Var._operator_dispatcher.or_(self, other)

    def __xor__(self, other) -> "Var":
        return Var._operator_dispatcher.xor(self, other)

    def __invert__(self) -> "Var":
        return Var._operator_dispatcher.not_(self)

    def __radd__(self, other) -> "Var":
        return Var._operator_dispatcher.add(other, self)

    def __rsub__(self, other) -> "Var":
        return Var._operator_dispatcher.sub(other, self)

    def __rmul__(self, other) -> "Var":
        return Var._operator_dispatcher.mul(other, self)

    def __rtruediv__(self, other) -> "Var":
        return Var._operator_dispatcher.truediv(other, self)

    def __rfloordiv__(self, other) -> "Var":
        return Var._operator_dispatcher.floordiv(other, self)

    def __rand__(self, other) -> "Var":
        return Var._operator_dispatcher.and_(other, self)

    def __ror__(self, other) -> "Var":
        return Var._operator_dispatcher.or_(other, self)

    def __rxor__(self, other) -> "Var":
        return Var._operator_dispatcher.xor(other, self)


def result_type(
    *types: Union[Var, numpy.generic, int, float],
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
