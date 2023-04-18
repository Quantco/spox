"""Module containing experimental Spox features that may be standard in the future."""

from contextlib import contextmanager
from typing import Iterable, List, Optional, Union

import numpy as np
import numpy.typing as npt

import spox._node
import spox._value_prop
from spox._graph import initializer as _initializer
from spox._type_system import Tensor
from spox._var import Var

TypeWarningLevel = spox._node.TypeWarningLevel


def set_type_warning_level(level: TypeWarningLevel) -> None:
    spox._node._TYPE_WARNING_LEVEL = level


@contextmanager
def type_warning_level(level: TypeWarningLevel):
    prev_level = spox._node._TYPE_WARNING_LEVEL
    set_type_warning_level(level)
    yield
    set_type_warning_level(prev_level)


ValuePropBackend = spox._value_prop.ValuePropBackend


def set_value_prop_backend(backend: ValuePropBackend) -> None:
    spox._value_prop._VALUE_PROP_BACKEND = backend


@contextmanager
def value_prop_backend(backend: ValuePropBackend):
    prev_backend = spox._value_prop._VALUE_PROP_BACKEND
    set_value_prop_backend(backend)
    yield
    set_value_prop_backend(prev_backend)


def initializer(value: npt.ArrayLike, dtype: npt.DTypeLike = None) -> Var:
    """
    Create a Var with a constant value.

    Parameters
    ----------
    value
        Array-like value for the variable.
    dtype
        Data type for the given value. If ``None``, it is inferred from the value
        using numpy rules (``numpy.array(value)``).

    Returns
    -------
    Var
        Variable with the given constant ``value``.

    Notes
    -----
    When the model is built, constants created by this function become initializers.
    As such, they are independent of an opset version and are listed separately
    in the model. Initializers are also used internally in Spox.
    """
    return _initializer(np.array(value, dtype))


class _NumpyLikeOperatorDispatcher:
    def __init__(self, op, type_promotion: bool, constant_promotion: bool):
        self.op = op
        self.type_promotion = type_promotion
        self.constant_promotion = constant_promotion

    def _promote(
        self, *args: Union[Var, np.generic, int, float], to_floating: bool = False
    ) -> Iterable[Optional[Var]]:
        """
        Apply constant promotion and type promotion to given parameters,
        creating constants and/or casting.
        """
        targets: List[Union[np.dtype, np.generic, int, float]] = [
            x.type.dtype if isinstance(x, Var) and isinstance(x.type, Tensor) else x
            for x in args
        ]
        if self.type_promotion:
            target_type = np.dtype(np.result_type(*targets))
            if to_floating and not issubclass(target_type.type, np.floating):
                target_type = np.float64
        else:
            dtypes = {dtype for dtype in targets if isinstance(dtype, np.dtype)}
            if len(dtypes) > 1:
                raise TypeError(
                    f"Inconsistent types for Var operator with no type promotion: {dtypes}."
                )
            (target_type,) = dtypes
            if issubclass(target_type.type, np.integer):
                disagreeing = [
                    value
                    for value in targets
                    if issubclass(np.result_type(value).type, np.floating)
                ]
                if disagreeing:
                    raise TypeError(
                        f"Floating constant operands promoting to integer type {target_type}: {disagreeing}"
                    )
            # TODO: Handle more constant-target inconsistencies here?

        def _promote_target(obj: Union[Var, np.generic, int, float]) -> Optional[Var]:
            if self.constant_promotion and isinstance(obj, (np.generic, int, float)):
                return self.op.const(np.array(obj, dtype=target_type))
            elif isinstance(obj, Var):
                return self.op.cast(obj, to=target_type) if self.type_promotion else obj
            raise TypeError(
                f"Bad value '{obj!r}' of type {type(obj).__name__!r} for operator overloading with Var. "
                f"({self.type_promotion=}, {self.constant_promotion=})"
            )

        return tuple(var for var in map(_promote_target, args))

    def add(self, a, b) -> Var:
        a, b = self._promote(a, b)
        return self.op.add(a, b)

    def sub(self, a, b) -> Var:
        a, b = self._promote(a, b)
        return self.op.sub(a, b)

    def mul(self, a, b) -> Var:
        a, b = self._promote(a, b)
        return self.op.mul(a, b)

    def truediv(self, a, b) -> Var:
        a, b = self._promote(a, b, to_floating=True)
        return self.op.div(a, b)

    def floordiv(self, a, b) -> Var:
        a, b = self._promote(a, b)
        c = self.op.div(a, b)
        if isinstance(c.type, Tensor) and not issubclass(c.type._elem_type, np.integer):
            c = self.op.floor(c)
        return c

    def neg(self, a: Var) -> Var:
        return self.op.neg(a)

    def and_(self, a: Var, b: Var) -> Var:
        return self.op.and_(a, b)

    def or_(self, a: Var, b: Var) -> Var:
        return self.op.or_(a, b)

    def xor(self, a: Var, b: Var) -> Var:
        return self.op.xor(a, b)

    def not_(self, a: Var) -> Var:
        return self.op.not_(a)


@contextmanager
def operator_overloading(
    op, type_promotion: bool = False, constant_promotion: bool = True
):
    """Enable operator overloading on Var for this block.

    May be used either as a context manager, or a decorator.

    Semantics of operator overloading attempt to follow usual rules of Python and numpy.
    By default, type promotion is not enabled.

    Parameters
    ----------
    op
        An opset module of domain ``ai.onnx`` that will be used to construct the operator dispatcher.
        It's recommended to use a modern version, like 17, and rely on version converters.
    type_promotion
        Whether operator overloading should implicitly promote types according to numpy rules.
        Modifies the behaviour of true division to cast to ``float64`` before the operation
        if the type was not conclusively floating (as in numpy).
        False by default.
    constant_promotion
        Whether operator overloading should implicitly promote primitive scalar constants to Var.
        True by default.

    Examples
    --------
    >>> import numpy as np
    >>> from spox.opset.ai.onnx import v17 as op
    >>> x, y = op.const(2), op.const(3)
    >>> with operator_overloading(op):
    ...     z = x + y
    >>> assert z._get_value() == np.array(5)
    >>> @operator_overloading(op)
    ... def foo():
    ...    return x * y
    >>> assert foo()._get_value() == np.array(6)
    """
    prev_dispatcher = Var._operator_dispatcher
    Var._operator_dispatcher = _NumpyLikeOperatorDispatcher(
        op, type_promotion, constant_promotion
    )
    yield
    Var._operator_dispatcher = prev_dispatcher


__all__ = [
    # Type warning levels
    "TypeWarningLevel",
    "set_type_warning_level",
    "type_warning_level",
    # Initializer-backed constants
    "initializer",
    # Value propagation backend
    "ValuePropBackend",
    "set_value_prop_backend",
    "value_prop_backend",
    # Operator overloading on Var
    "operator_overloading",
]
